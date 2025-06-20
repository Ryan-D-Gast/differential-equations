//! Adaptive Runge-Kutta methods for DDEs

use super::{ExplicitRungeKutta, Delay, Adaptive};
use crate::{
    Error, Status,
    methods::h_init::InitialStepSize,
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    dde::{DDENumericalMethod, DDE},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};
use std::collections::VecDeque;

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData, const O: usize, const S: usize, const I: usize> DDENumericalMethod<L, T, V, H, D> for ExplicitRungeKutta<Delay, Adaptive, T, V, D, O, S, I> {
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &V, phi: &H) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>,
    {
        let mut evals = Evals::new();

        // Initialize solver state
        self.t0 = t0;
        self.t = t0;
        self.y = *y0;
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.status = Status::Initialized;
        self.steps = 0;
        self.stiffness_counter = 0;
        self.history = VecDeque::new();

        // Initialize arrays for lags and delayed states
        let mut lags = [T::zero(); L];
        let mut yd = [V::zeros(); L];

        // Evaluate initial lags and delayed states
        if L > 0 {
            dde.lags(self.t, &self.y, &mut lags);
            for i in 0..L {
                if lags[i] <= T::zero() {
                    return Err(Error::BadInput {
                        msg: "All lags must be positive.".to_string(),
                    });
                }
                let t_delayed = self.t - lags[i];
                // Ensure delayed time is within history range (t_delayed <= t0)
                if (t_delayed - t0) * (tf - t0).signum() > T::default_epsilon() {
                    return Err(Error::BadInput {
                        msg: format!(
                            "Initial delayed time {} is out of history range (t <= {}).",
                            t_delayed, t0
                        ),
                    });
                }
                yd[i] = phi(t_delayed);
            }
        }

        // Calculate initial derivative
        dde.diff(self.t, &self.y, &yd, &mut self.dydt);
        evals.fcn += 1;
        self.dydt_prev = self.dydt;        // Store initial state in history
        self.history.push_back((self.t, self.y, self.dydt));

        // Calculate initial step size h0 if not provided
        if self.h0 == T::zero() {
            // Adaptive step method
            self.h0 = InitialStepSize::<Delay>::compute(dde, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max, phi, &self.k[0], &mut evals);
            evals.fcn += 2; // h_init performs 2 function evaluations
        }

        // Validate and set initial step size h
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }
        Ok(evals)
    }

    fn step<F>(&mut self, dde: &F, phi: &H) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>,
    {
        let mut evals = Evals::new();

        // Validate step size
        if self.h.abs() < T::default_epsilon() {
            self.status = Status::Error(Error::StepSize { t: self.t, y: self.y });
            return Err(Error::StepSize { t: self.t, y: self.y });
        }

        // Check maximum number of steps
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps { t: self.t, y: self.y });
            return Err(Error::MaxSteps { t: self.t, y: self.y });
        }
        self.steps += 1;

        // Initialize variables for the step
        let mut lags = [T::zero(); L];
        let mut yd = [V::zeros(); L];

        // Store current derivative as k[0] for RK computations
        self.k[0] = self.dydt;

        // DDE: Determine if iterative approach for lag handling is needed
        let mut min_lag_abs = T::infinity();
        if L > 0 {
            // Predict y at t+h using Euler step to estimate lags at t+h
            let y_pred_for_lags = self.y + self.k[0] * self.h;
            dde.lags(self.t + self.h, &y_pred_for_lags, &mut lags);
            for i in 0..L {
                min_lag_abs = min_lag_abs.min(lags[i].abs());
            }
        }

        // If lag values have to be extrapolated, we need to iterate for convergence
        let max_iter: usize = if L > 0 && min_lag_abs < self.h.abs() && min_lag_abs > T::zero() {
            5
        } else {
            1
        };

        let mut y_next_candidate_iter = self.y; // Approximated y at t+h, refined in DDE iterations
        let mut dydt_next_candidate_iter = V::zeros(); // Derivative at t+h using y_next_candidate_iter
        let mut y_prev_candidate_iter = self.y; // y_next_candidate_iter from previous DDE iteration
        let mut dde_iteration_failed = false;
        let mut err_norm: T = T::zero(); // Error norm for step size control

        // DDE iteration loop (for handling implicit lags or just one pass for explicit)
        for iter_idx in 0..max_iter {
            if iter_idx > 0 {
                y_prev_candidate_iter = y_next_candidate_iter;
            }

            // Compute Runge-Kutta stages
            for i in 1..self.stages {
                let mut y_stage = self.y;
                for j in 0..i {
                    y_stage += self.k[j] * (self.a[i][j] * self.h);
                }
                // Evaluate delayed states for the current stage
                if L > 0 {
                    dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut lags);
                    self.lagvals(self.t + self.c[i] * self.h, &lags, &mut yd, phi);
                }
                dde.diff(self.t + self.c[i] * self.h, &y_stage, &yd, &mut self.k[i]);
            }
            evals.fcn += self.stages - 1; // k[0] was already available

            // Adaptive methods: compute high and low order solutions for error estimation
            let mut y_high = self.y; // Higher order solution
            for i in 0..self.stages {
                y_high += self.k[i] * (self.b[i] * self.h);
            }
            let mut y_low = self.y; // Lower order solution (for error estimation)
            if let Some(bh_coeffs) = &self.bh {
                for i in 0..self.stages {
                    y_low += self.k[i] * (bh_coeffs[i] * self.h);
                }
            }
            let err_vec: V = y_high - y_low; // Error vector

            // Calculate error norm (||err||)
            err_norm = T::zero();
            for n in 0..self.y.len() {
                let tol = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
                err_norm = err_norm.max((err_vec.get(n) / tol).abs());
            }

            // DDE iteration convergence check (if max_iter > 1)
            if max_iter > 1 && iter_idx > 0 {
                let mut dde_iteration_error = T::zero();
                let n_dim = self.y.len();
                for i_dim in 0..n_dim {
                    let scale = self.atol + self.rtol * y_prev_candidate_iter.get(i_dim).abs().max(y_high.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val = y_high.get(i_dim) - y_prev_candidate_iter.get(i_dim);
                        dde_iteration_error += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    dde_iteration_error = (dde_iteration_error / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if dde_iteration_error <= self.rtol * T::from_f64(0.1).unwrap() {
                    break; // DDE iteration converged
                }
                if iter_idx == max_iter - 1 { // Last iteration
                    dde_iteration_failed = dde_iteration_error > self.rtol * T::from_f64(0.1).unwrap();
                }
            }
            y_next_candidate_iter = y_high; // Update candidate solution for t+h

            // Compute derivative at t+h with the current candidate y_next_candidate_iter
            if L > 0 {
                dde.lags(self.t + self.h, &y_next_candidate_iter, &mut lags);
                self.lagvals(self.t + self.h, &lags, &mut yd, phi);
            }
            dde.diff(self.t + self.h, &y_next_candidate_iter, &yd, &mut dydt_next_candidate_iter);
            evals.fcn += 1;
        } // End of DDE iteration loop

        // Handle DDE iteration failure: reduce step size and retry
        if dde_iteration_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            // Ensure step size is not smaller than a fraction of the minimum lag, if applicable
            if L > 0 && min_lag_abs > T::zero() && self.h.abs() < T::from_f64(2.0).unwrap() * min_lag_abs {
                self.h = min_lag_abs * sign; // Or some factor of min_lag_abs
            }
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep; // Indicate step rejection due to DDE iteration
            // self.k[0] = self.dydt; // k[0] is already self.dydt, no need to reset
            return Ok(evals); // Return to retry step with smaller h
        }

        // Step size controller for adaptive methods (based on err_norm)
        let order_t = T::from_usize(self.order).unwrap();
        let err_order_inv = T::one() / order_t;
        let mut scale_factor = self.safety_factor * err_norm.powf(-err_order_inv);
        scale_factor = scale_factor.max(self.min_scale).min(self.max_scale);
        
        let h_new = self.h * scale_factor;

        // Step acceptance/rejection logic
        if err_norm <= T::one() { // Step accepted
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.dydt; // Derivative at t_prev
            self.h_prev = self.h; // Store accepted step size

            if let Status::RejectedStep = self.status { // If previous step was rejected
                self.stiffness_counter = 0;
            }
            self.status = Status::Solving;

            // Compute additional stages for dense output if available
            if self.bi.is_some() {
                for i in 0..(I - S) { // I is total stages, S is main method stages
                    let mut y_stage_dense = self.y; // Base for dense stage calculation
                    // Sum up contributions from previous k values for this dense stage
                    for j in 0..self.stages + i { // self.stages is S
                        y_stage_dense += self.k[j] * (self.a[self.stages + i][j] * self.h);
                    }
                    // Evaluate lags and derivative for the dense stage
                    if L > 0 {
                        dde.lags(self.t + self.c[self.stages + i] * self.h, &y_stage_dense, &mut lags);
                        self.lagvals(self.t + self.c[self.stages + i] * self.h, &lags, &mut yd, phi);
                    }
                    dde.diff(self.t + self.c[self.stages + i] * self.h, &y_stage_dense, &yd, &mut self.k[self.stages + i]);
                }
                evals.fcn += I - S; // Account for function evaluations for dense stages
            }

            // Update state to t + h
            self.t += self.h;
            self.y = y_next_candidate_iter;
            // Update derivative at the new state (self.t, self.y)
            if L > 0 {
                dde.lags(self.t, &self.y, &mut lags);
                self.lagvals(self.t, &lags, &mut yd, phi);
            }
            dde.diff(self.t, &self.y, &yd, &mut self.dydt); // This is f(t_new, y_new)
            evals.fcn += 1;

            // Update continuous output buffer and remove old entries if max_delay is set
            self.history.push_back((self.t, self.y, self.dydt));
            if let Some(max_delay) = self.max_delay {
                let cutoff_time = self.t - max_delay;
                while let Some((t_front, _, _)) = self.history.get(1){
                    if *t_front < cutoff_time {
                        self.history.pop_front();
                    } else {
                        break; // Stop pruning when we reach the cutoff time
                    }
                }
            }

            self.h = constrain_step_size(h_new, self.h_min, self.h_max); // Set next step size
        } else { // Step rejected
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            // Check for excessive rejections (potential stiffness)
            if self.stiffness_counter >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness { t: self.t, y: self.y });
                return Err(Error::Stiffness { t: self.t, y: self.y });
            }
            // Reduce step size for next attempt
            self.h = constrain_step_size(h_new, self.h_min, self.h_max);
        }
        Ok(evals)
    }

    fn t(&self) -> T { self.t }
    fn y(&self) -> &V { &self.y }
    fn t_prev(&self) -> T { self.t_prev }
    fn y_prev(&self) -> &V { &self.y_prev }
    fn h(&self) -> T { self.h }
    fn set_h(&mut self, h: T) { self.h = h; }
    fn status(&self) -> &Status<T, V, D> { &self.status }
    fn set_status(&mut self, status: Status<T, V, D>) { self.status = status; }
}

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> ExplicitRungeKutta<Delay, Adaptive, T, V, D, O, S, I> {    
    fn lagvals<const L: usize, H>(&mut self, t_stage: T, lags: &[T; L], yd: &mut [V; L], phi: &H) 
    where 
        H: Fn(T) -> V,
    {
        for i in 0..L {
            let t_delayed = t_stage - lags[i];
            
            // Check if delayed time falls within the history period (t_delayed <= t0)
            if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                yd[i] = phi(t_delayed);
            // If t_delayed is after t_prev then use interpolation function
            } else if (t_delayed - self.t_prev) * self.h.signum() > T::default_epsilon() {
                if self.bi.is_some() {
                    let s = (t_delayed - self.t_prev) / self.h_prev;
                    
                    let bi_coeffs = self.bi.as_ref().unwrap();

                    let mut cont = [T::zero(); I];
                    for i in 0..I {
                        if i < self.cont.len() && i < bi_coeffs.len() {
                            cont[i] = bi_coeffs[i][self.dense_stages - 1];
                            for j in (0..self.dense_stages - 1).rev() {
                                cont[i] = cont[i] * s + bi_coeffs[i][j];
                            }
                            cont[i] *= s;
                        }
                    }

                    let mut y_interp = self.y_prev;
                    for i in 0..I {
                        if i < self.k.len() && i < self.cont.len() {
                            y_interp += self.k[i] * (cont[i] * self.h_prev);
                        }
                    }
                    yd[i] = y_interp;
                } else {
                    yd[i] = cubic_hermite_interpolate(
                        self.t_prev, 
                        self.t, 
                        &self.y_prev, 
                        &self.y, 
                        &self.dydt_prev, 
                        &self.dydt, 
                        t_delayed
                    );
                }            // If t_delayed is before t_prev and after t0, we need to search in the history
            } else {                // Search through history to find appropriate interpolation points
                let mut found_interpolation = false;
                let buffer = &self.history;
                // Find two consecutive points that sandwich t_delayed using iterators
                let mut buffer_iter = buffer.iter();
                if let Some(mut prev_entry) = buffer_iter.next() {
                    for curr_entry in buffer_iter {
                        let (t_left, y_left, dydt_left) = prev_entry;
                        let (t_right, y_right, dydt_right) = curr_entry;
                        
                        // Check if t_delayed is between these two points
                        let is_between = if self.h.signum() > T::zero() {
                            // Forward integration: t_left <= t_delayed <= t_right
                            *t_left <= t_delayed && t_delayed <= *t_right
                        } else {
                            // Backward integration: t_right <= t_delayed <= t_left
                            *t_right <= t_delayed && t_delayed <= *t_left
                        };
                        
                        if is_between {
                            // Use cubic Hermite interpolation between these points
                            yd[i] = cubic_hermite_interpolate(
                                *t_left,
                                *t_right,
                                y_left,
                                y_right,
                                dydt_left,
                                dydt_right,
                                t_delayed
                            );
                            found_interpolation = true;
                            break;
                        }
                        prev_entry = curr_entry;
                    }
                }// If not found in history, this indicates insufficient history in buffer
                if !found_interpolation {
                    // Debug: show buffer contents
                    let buffer = &self.history;
                    println!("Buffer contents ({} entries):", buffer.len());
                    for (idx, (t_buf, _, _)) in buffer.iter().enumerate() {
                        if idx < 5 || idx >= buffer.len() - 5 {
                            println!("  [{}] t = {}", idx, t_buf);
                        } else if idx == 5 {
                            println!("  ... ({} more entries) ...", buffer.len() - 10);
                        }
                    }
                    panic!("Insufficient history in history for t_delayed = {} (t_prev = {}, t = {}). Buffer may need to retain more points or there's a logic error in determining interpolation intervals.", t_delayed, self.t_prev, self.t);
                }
            }
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Delay, Adaptive, T, V, D, O, S, I> {
    /// Interpolates the solution at a given time `t_interp`.
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        let posneg = (self.t - self.t_prev).signum();
        if (t_interp - self.t_prev) * posneg < T::zero() || (t_interp - self.t) * posneg > T::zero() {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // If method has dense output coefficients, use them
        if self.bi.is_some() {
            // Calculate the normalized distance within the step [0, 1]
            let s = (t_interp - self.t_prev) / self.h_prev;
            
            // Get the interpolation coefficients
            let bi = self.bi.as_ref().unwrap();

            let mut cont = [T::zero(); I];
            // Compute the interpolation coefficients using Horner's method
            for i in 0..self.dense_stages {
                // Start with the highest-order term
                cont[i] = bi[i][self.order - 1];

                // Apply Horner's method
                for j in (0..self.order - 1).rev() {
                    cont[i] = cont[i] * s + bi[i][j];
                }

                // Multiply by s
                cont[i] *= s;
            }

            // Compute the interpolated value
            let mut y_interp = self.y_prev;
            for i in 0..I {
                y_interp += self.k[i] * cont[i] * self.h_prev;
            }

            Ok(y_interp)
        } else {
            // Otherwise use cubic Hermite interpolation
            let y_interp = cubic_hermite_interpolate(
                self.t_prev, 
                self.t, 
                &self.y_prev, 
                &self.y, 
                &self.dydt_prev, 
                &self.dydt, 
                t_interp
            );

            Ok(y_interp)
        }
    }
}