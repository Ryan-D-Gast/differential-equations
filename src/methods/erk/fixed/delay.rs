//! Fixed Runge-Kutta methods for DDEs

use crate::{
    Error, Status,
    dde::{DDE, DelayNumericalMethod},
    interpolate::{Interpolation, cubic_hermite_interpolate},
    methods::{Delay, ExplicitRungeKutta, Fixed},
    stats::Evals,
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};
use std::collections::VecDeque;

impl<
    const L: usize,
    T: Real,
    Y: State<T>,
    H: Fn(T) -> Y,
    D: CallBackData,
    const O: usize,
    const S: usize,
    const I: usize,
> DelayNumericalMethod<L, T, Y, H, D> for ExplicitRungeKutta<Delay, Fixed, T, Y, D, O, S, I>
{
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &Y, phi: &H) -> Result<Evals, Error<T, Y>>
    where
        F: DDE<L, T, Y, D>,
    {
        // Initialize solver state
        let mut evals = Evals::new();
        self.t0 = t0;
        self.t = t0;
        self.y = *y0;
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.status = Status::Initialized;
        self.steps = 0;
        self.history = VecDeque::new();

        // Initialize arrays for lags and delayed states
        let mut lags = [T::zero(); L];
        let mut yd = [Y::zeros(); L];

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
        evals.function += 1;
        self.dydt_prev = self.dydt; // Store initial state in history
        self.history.push_back((self.t, self.y, self.dydt));

        // Calculate initial step size h0 if not provided
        if self.h0 == T::zero() {
            // Simple default step size for fixed-step methods
            let duration = (tf - t0).abs();
            let default_steps = T::from_usize(100).unwrap();
            self.h0 = duration / default_steps;
        }

        // Validate and set initial step size h
        match validate_step_size_parameters::<T, Y, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }
        Ok(evals)
    }

    fn step<F>(&mut self, dde: &F, phi: &H) -> Result<Evals, Error<T, Y>>
    where
        F: DDE<L, T, Y, D>,
    {
        let mut evals = Evals::new();

        // Check maximum number of steps
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
        }
        self.steps += 1;

        // Initialize variables for the step
        let mut lags = [T::zero(); L];
        let mut yd = [Y::zeros(); L];

        // Store current derivative as k[0] for RK computations
        self.k[0] = self.dydt; // DDE: Determine if iterative approach for lag handling is needed
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
        let mut dydt_next_candidate_iter = Y::zeros(); // Derivative at t+h using y_next_candidate_iter
        let mut y_prev_candidate_iter = self.y; // y_next_candidate_iter from previous DDE iteration
        let mut dde_iteration_failed = false;

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
            evals.function += self.stages - 1; // k[0] was already available

            // Compute solution
            let mut y_next = self.y;
            for i in 0..self.stages {
                y_next += self.k[i] * (self.b[i] * self.h);
            }

            // DDE iteration convergence check (if max_iter > 1)
            if max_iter > 1 && iter_idx > 0 {
                let mut dde_iteration_error = T::zero();
                let n_dim = self.y.len();
                for i_dim in 0..n_dim {
                    let scale = T::from_f64(1e-10).unwrap()
                        + y_prev_candidate_iter
                            .get(i_dim)
                            .abs()
                            .max(y_next.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val = y_next.get(i_dim) - y_prev_candidate_iter.get(i_dim);
                        dde_iteration_error += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    dde_iteration_error =
                        (dde_iteration_error / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if dde_iteration_error <= T::from_f64(1e-6).unwrap() {
                    break; // DDE iteration converged
                }
                if iter_idx == max_iter - 1 {
                    // Last iteration
                    dde_iteration_failed = dde_iteration_error > T::from_f64(1e-6).unwrap();
                }
            }
            y_next_candidate_iter = y_next; // Update candidate solution for t+h

            // Compute derivative at t+h with the current candidate y_next_candidate_iter
            if L > 0 {
                dde.lags(self.t + self.h, &y_next_candidate_iter, &mut lags);
                self.lagvals(self.t + self.h, &lags, &mut yd, phi);
            }
            dde.diff(
                self.t + self.h,
                &y_next_candidate_iter,
                &yd,
                &mut dydt_next_candidate_iter,
            );
            evals.function += 1;
        } // End of DDE iteration loop

        // Handle DDE iteration failure: reduce step size and retry
        if dde_iteration_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            // Ensure step size is not smaller than a fraction of the minimum lag, if applicable
            if L > 0
                && min_lag_abs > T::zero()
                && self.h.abs() < T::from_f64(2.0).unwrap() * min_lag_abs
            {
                self.h = min_lag_abs * sign; // Or some factor of min_lag_abs
            }
            self.status = Status::RejectedStep; // Indicate step rejection due to DDE iteration
            return Ok(evals); // Return to retry step with smaller h
        }

        // Store current state before update for interpolation
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Update state to t + h
        self.t += self.h;
        self.y = y_next_candidate_iter;

        // Calculate new derivative for next step
        if self.fsal {
            // If FSAL (First Same As Last) is enabled, we can reuse the last derivative
            self.dydt = self.k[S - 1];
        } else {
            // Otherwise, compute the new derivative
            if L > 0 {
                dde.lags(self.t, &self.y, &mut lags);
                self.lagvals(self.t, &lags, &mut yd, phi);
            }
            dde.diff(self.t, &self.y, &yd, &mut self.dydt);
            evals.function += 1;
        }

        // Compute additional stages for dense output if available
        if self.bi.is_some() {
            for i in 0..(I - S) {
                // I is total stages, S is main method stages
                let mut y_stage_dense = self.y_prev; // Use previous state as base
                // Sum up contributions from previous k values for this dense stage
                for j in 0..self.stages + i {
                    // self.stages is S
                    y_stage_dense += self.k[j] * (self.a[self.stages + i][j] * self.h);
                }
                // Evaluate lags and derivative for the dense stage
                if L > 0 {
                    dde.lags(
                        self.t_prev + self.c[self.stages + i] * self.h,
                        &y_stage_dense,
                        &mut lags,
                    );
                    self.lagvals(
                        self.t_prev + self.c[self.stages + i] * self.h,
                        &lags,
                        &mut yd,
                        phi,
                    );
                }
                dde.diff(
                    self.t_prev + self.c[self.stages + i] * self.h,
                    &y_stage_dense,
                    &yd,
                    &mut self.k[self.stages + i],
                );
            }
            evals.function += I - S; // Account for function evaluations for dense stages
        }

        // Update continuous output buffer and remove old entries if max_delay is set
        self.history.push_back((self.t, self.y, self.dydt));
        if let Some(max_delay) = self.max_delay {
            let cutoff_time = self.t - max_delay;
            while let Some((t_front, _, _)) = self.history.get(1) {
                if *t_front < cutoff_time {
                    self.history.pop_front();
                } else {
                    break; // Stop pruning when we reach the cutoff time
                }
            }
        }

        self.status = Status::Solving;
        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }
    fn y(&self) -> &Y {
        &self.y
    }
    fn t_prev(&self) -> T {
        self.t_prev
    }
    fn y_prev(&self) -> &Y {
        &self.y_prev
    }
    fn h(&self) -> T {
        self.h
    }
    fn set_h(&mut self, h: T) {
        self.h = h;
    }
    fn status(&self) -> &Status<T, Y, D> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y, D>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    ExplicitRungeKutta<Delay, Fixed, T, Y, D, O, S, I>
{
    pub fn lagvals<const L: usize, H>(
        &mut self,
        t_stage: T,
        lags: &[T; L],
        yd: &mut [Y; L],
        phi: &H,
    ) where
        H: Fn(T) -> Y,
    {
        for i in 0..L {
            let t_delayed = t_stage - lags[i];

            // Check if delayed time falls within the history period (t_delayed <= t0)
            if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                yd[i] = phi(t_delayed);
            // If t_delayed is after t_prev then use interpolation function
            } else if (t_delayed - self.t_prev) * self.h.signum() > T::default_epsilon() {
                if self.bi.is_some() {
                    let s = (t_delayed - self.t_prev) / self.h;

                    let bi_coeffs = self.bi.as_ref().unwrap();

                    let mut cont = [T::zero(); I];
                    for i in 0..I {
                        if i < cont.len() && i < bi_coeffs.len() {
                            cont[i] = bi_coeffs[i][self.dense_stages - 1];
                            for j in (0..self.dense_stages - 1).rev() {
                                cont[i] = cont[i] * s + bi_coeffs[i][j];
                            }
                            cont[i] *= s;
                        }
                    }

                    let mut y_interp = self.y_prev;
                    for i in 0..I {
                        if i < self.k.len() && i < cont.len() {
                            y_interp += self.k[i] * (cont[i] * self.h);
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
                        t_delayed,
                    );
                } // If t_delayed is before t_prev and after t0, we need to search in the history
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
                            // Forward integration: t_left <= t_delayed <= t_right
                            *t_left <= t_delayed && t_delayed <= *t_right
                        } else {
                            // Backward integration: t_right <= t_delayed <= t_left
                            *t_right <= t_delayed && t_delayed <= *t_left
                        };

                        if is_between {
                            // Use cubic Hermite interpolation between these points
                            yd[i] = cubic_hermite_interpolate(
                                *t_left, *t_right, y_left, y_right, dydt_left, dydt_right,
                                t_delayed,
                            );
                            found_interpolation = true;
                            break;
                        }
                        prev_entry = curr_entry;
                    }
                } // If not found in history, this indicates insufficient history in buffer
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
                    panic!(
                        "Insufficient history in history for t_delayed = {} (t_prev = {}, t = {}). Buffer may need to retain more points or there's a logic error in determining interpolation intervals.",
                        t_delayed, self.t_prev, self.t
                    );
                }
            }
        }
    }
}

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Interpolation<T, Y> for ExplicitRungeKutta<Delay, Fixed, T, Y, D, O, S, I>
{
    /// Interpolates the solution at a given time `t_interp`.
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        let posneg = self.h.signum();
        if (t_interp - self.t_prev) * posneg < T::zero() || (t_interp - self.t) * posneg > T::zero()
        {
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
                t_interp,
            );

            Ok(y_interp)
        }
    }
}
