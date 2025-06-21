//! Dormand-Prince Runge-Kutta methods for DDEs

use super::{ExplicitRungeKutta, Delay, DormandPrince};
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

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData, const O: usize, const S: usize, const I: usize> DDENumericalMethod<L, T, V, H, D> for ExplicitRungeKutta<Delay, DormandPrince, T, V, D, O, S, I> {
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
        self.status = Status::Initialized;        self.steps = 0;
        self.stiffness_counter = 0;
        self.non_stiffness_counter = 0;
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
                            "Delayed time {} is beyond initial time {}",
                            t_delayed, t0
                        ),
                    });
                }
                yd[i] = phi(t_delayed);
            }
        }

        // Calculate initial derivative
        dde.diff(self.t, &self.y, &yd, &mut self.k[0]);
        self.dydt = self.k[0];
        evals.fcn += 1;
        self.dydt_prev = self.dydt;

        // Store initial state in history
        self.history.push_back((self.t, self.y, self.dydt));

        // Calculate initial step size h0 if not provided
        if self.h0 == T::zero() {
            // Use Dormand-Prince specific step size calculation  
            self.h0 = InitialStepSize::<Delay>::compute(dde, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max, phi, &self.k[0], &mut evals);
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
        };        let mut y_next_candidate_iter = self.y; // Approximated y at t+h, refined in DDE iterations
        let mut y_prev_candidate_iter = self.y; // y_next_candidate_iter from previous DDE iteration
        let mut dde_iteration_failed = false;
        let mut err: T = T::zero(); // Error norm for step size control
        let mut ysti = V::zeros(); // Store last stage for stiffness detection

        // DDE iteration loop (for handling implicit lags or just one pass for explicit)
        for iter_idx in 0..max_iter {
            if iter_idx > 0 {
                y_prev_candidate_iter = y_next_candidate_iter;
            }

            // Compute Runge-Kutta stages
            let mut y_stage = V::zeros();
            for i in 1..self.stages {
                y_stage = V::zeros();
                for j in 0..i {
                    y_stage += self.k[j] * self.a[i][j];
                }
                y_stage = self.y + y_stage * self.h;

                // Evaluate delayed states for the current stage
                if L > 0 {
                    dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut lags);
                    self.lagvals(self.t + self.c[i] * self.h, &lags, &mut yd, phi);
                }
                dde.diff(self.t + self.c[i] * self.h, &y_stage, &yd, &mut self.k[i]);
            }
            evals.fcn += self.stages - 1; // k[0] was already available

            // Store the last stage for stiffness detection
            ysti = y_stage;

            // Calculate the line segment for the new y value
            let mut yseg = V::zeros();
            for i in 0..self.stages {
                yseg += self.k[i] * self.b[i];
            }

            // Calculate the new y value using the line segment
            let y_new = self.y + yseg * self.h;

            // Dormand-Prince error estimation
            let er = self.er.unwrap();
            let n = self.y.len();
            let mut err_val = T::zero();
            let mut err2 = T::zero();
            let mut erri;
            for i in 0..n {
                // Calculate the error scale
                let sk = self.atol + self.rtol * self.y.get(i).abs().max(y_new.get(i).abs());

                // Primary error term
                erri = T::zero();
                for j in 0..self.stages {
                    erri += er[j] * self.k[j].get(i);
                }
                err_val += (erri / sk).powi(2);

                // Optional secondary error term
                if let Some(bh) = &self.bh {
                    erri = yseg.get(i);
                    for j in 0..self.stages {
                        erri -= bh[j] * self.k[j].get(i);
                    }
                    err2 += (erri / sk).powi(2);
                }
            }
            let mut deno = err_val + T::from_f64(0.01).unwrap() * err2;
            if deno <= T::zero() {
                deno = T::one();
            }
            err = self.h.abs() * err_val * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();

            // DDE iteration convergence check (if max_iter > 1)
            if max_iter > 1 && iter_idx > 0 {
                let mut dde_iteration_error = T::zero();
                let n_dim = self.y.len();
                for i_dim in 0..n_dim {
                    let scale = self.atol + self.rtol * y_prev_candidate_iter.get(i_dim).abs().max(y_new.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val = y_new.get(i_dim) - y_prev_candidate_iter.get(i_dim);
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
            y_next_candidate_iter = y_new; // Update candidate solution for t+h

            // Store ysti for potential stiffness detection
            if iter_idx == max_iter - 1 || max_iter == 1 {
                // Keep ysti from the final iteration for stiffness detection
            }
        } // End of DDE iteration loop

        // Handle DDE iteration failure: reduce step size and retry
        if dde_iteration_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            // Ensure step size is not smaller than a fraction of the minimum lag, if applicable
            if L > 0 && min_lag_abs > T::zero() && self.h.abs() < T::from_f64(2.0).unwrap() * min_lag_abs {
                self.h = min_lag_abs * sign;
            }
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
            return Ok(evals); // Return to retry step with smaller h
        }

        // Step acceptance/rejection logic
        if err <= T::one() { // Step accepted
            let y_new = y_next_candidate_iter;
            let t_new = self.t + self.h;

            // Calculate the new derivative at the new point
            if L > 0 {
                dde.lags(t_new, &y_new, &mut lags);
                self.lagvals(t_new, &lags, &mut yd, phi);
            }
            dde.diff(t_new, &y_new, &yd, &mut self.dydt);
            evals.fcn += 1;            // Stiffness detection (every 100 steps)
            let n_stiff_threshold = 100;
            if self.steps % n_stiff_threshold == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = {
                    let mut yseg = V::zeros();
                    for i in 0..self.stages {
                        yseg += self.k[i] * self.b[i];
                    }
                    yseg - self.k[S-1]
                };
                for i in 0..sqr.len() {
                    stdnum += sqr.get(i).powi(2);
                }
                let sqr = self.dydt - ysti;
                for i in 0..sqr.len() {
                    stden += sqr.get(i).powi(2);
                }

                if stden > T::zero() {
                    let h_lamb = self.h * (stdnum / stden).sqrt();
                    if h_lamb > T::from_f64(6.1).unwrap() {
                        self.non_stiffness_counter = 0;
                        self.stiffness_counter += 1;
                        if self.stiffness_counter == 15 {
                            self.status = Status::Error(Error::Stiffness {
                                t: self.t,
                                y: self.y,
                            });
                            return Err(Error::Stiffness {
                                t: self.t,
                                y: self.y,
                            });
                        }
                    }
                } else {
                    self.non_stiffness_counter += 1;
                    if self.non_stiffness_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }

            // Preparation for dense output / interpolation
            self.cont[0] = self.y;
            let ydiff = y_new - self.y;
            self.cont[1] = ydiff;
            let bspl = self.k[0] * self.h - ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.dydt * self.h - bspl;

            // If method has dense output stages, compute them
            if let Some(bi) = &self.bi {
                // Compute extra stages for dense output
                if I > S {
                    // First dense output coefficient, k{i=order+1}, is the derivative at the new point
                    self.k[self.stages] = self.dydt;                    for i in S+1..I {
                        let mut y_stage = V::zeros();
                        for j in 0..i {
                            y_stage += self.k[j] * self.a[i][j];
                        }
                        y_stage = self.y + y_stage * self.h;

                        if L > 0 {
                            dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut lags);
                            // Manually inline the lagvals logic to avoid borrowing conflicts
                            for lag_idx in 0..L {
                                let t_delayed = (self.t + self.c[i] * self.h) - lags[lag_idx];
                                
                                // Check if delayed time falls within the history period (t_delayed <= t0)
                                if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                                    yd[lag_idx] = phi(t_delayed);
                                // If t_delayed is after t_prev then use interpolation function
                                } else if (t_delayed - self.t_prev) * self.h.signum() > T::default_epsilon() {
                                    if self.bi.is_some() {
                                        let s = (t_delayed - self.t_prev) / self.h_prev;
                                        let s1 = T::one() - s;        
                                        let ilast = self.cont.len() - 1;
                                        let poly = (1..ilast).rev().fold(self.cont[ilast], |acc, cont_i| {            
                                            let factor = if cont_i >= 4 {
                                                if (ilast - cont_i) % 2 == 1 { s1 } else { s }
                                            } else {
                                                if cont_i % 2 == 1 { s1 } else { s }
                                            };
                                            acc * factor + self.cont[cont_i]
                                        });
                                        yd[lag_idx] = self.cont[0] + poly * s;
                                    } else {
                                        yd[lag_idx] = cubic_hermite_interpolate(
                                            self.t_prev, 
                                            self.t, 
                                            &self.y_prev, 
                                            &self.y, 
                                            &self.dydt_prev, 
                                            &self.dydt, 
                                            t_delayed
                                        );
                                    }
                                } else {
                                    // Search through history to find appropriate interpolation points
                                    let mut found_interpolation = false;
                                    let buffer = &self.history;
                                    let mut buffer_iter = buffer.iter();
                                    if let Some(mut prev_entry) = buffer_iter.next() {
                                        for curr_entry in buffer_iter {
                                            let (t_left, y_left, dydt_left) = prev_entry;
                                            let (t_right, y_right, dydt_right) = curr_entry;
                                            
                                            let is_between = if self.h.signum() > T::zero() {
                                                *t_left <= t_delayed && t_delayed <= *t_right
                                            } else {
                                                *t_right <= t_delayed && t_delayed <= *t_left
                                            };
                                            
                                            if is_between {
                                                yd[lag_idx] = cubic_hermite_interpolate(
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
                                    }
                                    if !found_interpolation {
                                        panic!("Insufficient history for t_delayed = {} (t_prev = {}, t = {})", t_delayed, self.t_prev, self.t);
                                    }
                                }
                            }
                        }
                        dde.diff(self.t + self.c[i] * self.h, &y_stage, &yd, &mut self.k[i]);
                        evals.fcn += 1;
                    }
                }

                // Compute dense output coefficients
                for i in 4..self.order {
                    self.cont[i] = V::zeros();
                    for j in 0..self.dense_stages {
                        self.cont[i] += self.k[j] * bi[i][j];
                    }
                    self.cont[i] = self.cont[i] * self.h;
                }
            }

            // For interpolation
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.k[0];
            self.h_prev = self.h;

            // Update state to t + h
            self.t = t_new;
            self.y = y_new;
            self.k[0] = self.dydt;

            // Update continuous output buffer and remove old entries if max_delay is set
            self.history.push_back((self.t, self.y, self.dydt));
            if let Some(max_delay) = self.max_delay {
                let cutoff_time = self.t - max_delay;
                while let Some((t_front, _, _)) = self.history.get(1){
                    if *t_front < cutoff_time {
                        self.history.pop_front();
                    } else {
                        break;
                    }
                }
            }            // Check if previous step is rejected
            if let Status::RejectedStep = self.status {
                self.status = Status::Solving;
            }
        } else {
            // Step Rejected
            self.status = Status::RejectedStep;
        }

        // Calculate new step size for adaptive methods
        let order = T::from_usize(self.order).unwrap();
        let err_order = T::one() / order;

        // Step size controller
        let scale = self.safety_factor * err.powf(-err_order);
        let scale = scale.max(self.min_scale).min(self.max_scale);
        self.h *= scale;

        // Ensure step size is within bounds
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);
        
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

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> ExplicitRungeKutta<Delay, DormandPrince, T, V, D, O, S, I> {    
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
                    
                    // Evaluate the interpolation polynomial at the requested time
                    let s1 = T::one() - s;        
                    
                    // Functional implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
                    let ilast = self.cont.len() - 1;
                    let poly = (1..ilast).rev().fold(self.cont[ilast], |acc, i| {            
                        let factor = if i >= 4 {
                            // For the higher-order part (conpar), alternate s and s1 based on index parity
                            if (ilast - i) % 2 == 1 { s1 } else { s }
                        } else {
                            // For the main polynomial part, pattern is [s1, s, s1] for indices [3, 2, 1]
                            if i % 2 == 1 { s1 } else { s }
                        };
                        acc * factor + self.cont[i]
                    });
                    
                    // Final multiplication by s for the outermost level
                    let y_interp = self.cont[0] + poly * s;
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
                }
            // If t_delayed is before t_prev and after t0, we need to search in the history
            } else {
                // Search through history to find appropriate interpolation points
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
                            *t_left <= t_delayed && t_delayed <= *t_right
                        } else {
                            *t_right <= t_delayed && t_delayed <= *t_left
                        };
                        
                        if is_between {
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
                }
                // If not found in history, this indicates insufficient history in buffer
                if !found_interpolation {
                    // Debug: show buffer contents
                    let buffer = &self.history;
                    println!("Buffer contents ({} entries):", buffer.len());
                    for (idx, (t_buf, _, _)) in buffer.iter().enumerate() {
                        println!("  [{}]: t = {}", idx, t_buf);
                    }
                    panic!("Insufficient history in history for t_delayed = {} (t_prev = {}, t = {}). Buffer may need to retain more points or there's a logic error in determining interpolation intervals.", t_delayed, self.t_prev, self.t);
                }
            }
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Delay, DormandPrince, T, V, D, O, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {        
        // Check if interpolation is out of bounds
        let posneg = (self.t - self.t_prev).signum();
        if (t_interp - self.t_prev) * posneg < T::zero() || (t_interp - self.t) * posneg > T::zero() {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }        
        
        // Evaluate the interpolation polynomial at the requested time
        let s = (t_interp - self.t_prev) / self.h_prev;
        let s1 = T::one() - s;        
        
        // Functional implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
        let ilast = self.cont.len() - 1;
        let poly = (1..ilast).rev().fold(self.cont[ilast], |acc, i| {            
            let factor = if i >= 4 {
                // For the higher-order part (conpar), alternate s and s1 based on index parity
                if (ilast - i) % 2 == 1 { s1 } else { s }
            } else {
                // For the main polynomial part, pattern is [s1, s, s1] for indices [3, 2, 1]
                if i % 2 == 1 { s1 } else { s }
            };
            acc * factor + self.cont[i]
        });
        
        // Final multiplication by s for the outermost level
        let y_interp = self.cont[0] + poly * s;

        Ok(y_interp)
    }
}
