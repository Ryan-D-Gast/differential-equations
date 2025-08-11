//! Fixed-step explicit Runge–Kutta methods for Delay Differential Equations (DDEs)

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

        // DDE requires at least one lag
        if L <= 0 {
            return Err(Error::NoLags);
        }
        self.t0 = t0;
        self.t = t0;
        self.y = *y0;
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.status = Status::Initialized;
        self.steps = 0;
        self.history = VecDeque::new();

        // Delay buffers
        let mut delays = [T::zero(); L];
        let mut y_delayed = [Y::zeros(); L];

        // Evaluate initial delays and history
        dde.lags(self.t, &self.y, &mut delays);
        for i in 0..L {
            let t_delayed = self.t - delays[i];
            // Ensure delayed time is within history range
            if (t_delayed - t0) * (tf - t0).signum() > T::default_epsilon() {
                return Err(Error::BadInput {
                    msg: format!(
                        "Initial delayed time {} is out of history range (t <= {}).",
                        t_delayed, t0
                    ),
                });
            }
            y_delayed[i] = phi(t_delayed);
        }

        // Initial derivative
        dde.diff(self.t, &self.y, &y_delayed, &mut self.dydt);
        evals.function += 1;
        self.dydt_prev = self.dydt; // Store initial state in history
        self.history.push_back((self.t, self.y, self.dydt));

        // Initial step size
        if self.h0 == T::zero() {
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

        // Step buffers
        let mut delays = [T::zero(); L];
        let mut y_delayed = [Y::zeros(); L];

        // Store current derivative as k[0] for RK computations
        // Seed k[0] with current derivative
        self.k[0] = self.dydt;
        let mut min_delay_abs = T::infinity();
        // Predict y(t+h) to estimate delays at t+h
        let y_pred_for_lags = self.y + self.k[0] * self.h;
        dde.lags(self.t + self.h, &y_pred_for_lags, &mut delays);
        for i in 0..L {
            min_delay_abs = min_delay_abs.min(delays[i].abs());
        }

        // Delay iteration count
    let max_iter: usize = if min_delay_abs < self.h.abs() && min_delay_abs > T::zero() {
            5
        } else {
            1
        };

        let mut y_next_candidate_iter = self.y; // Approximated y at t+h, refined in DDE iterations
        let mut dydt_next_candidate_iter = Y::zeros(); // Derivative at t+h using y_next_candidate_iter
        let mut y_prev_candidate_iter = self.y; // y_next_candidate_iter from previous DDE iteration
        let mut dde_iteration_failed = false;

        // DDE iteration loop
        for iter_idx in 0..max_iter {
            if iter_idx > 0 {
                y_prev_candidate_iter = y_next_candidate_iter;
            }

            // Compute stages
            for i in 1..self.stages {
                let mut y_stage = self.y;
                for j in 0..i {
                    y_stage += self.k[j] * (self.a[i][j] * self.h);
                }
                // Delayed states for this stage
                dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut delays);
                if let Err(e) = self.lagvals(self.t + self.c[i] * self.h, &delays, &mut y_delayed, phi) {
                    self.status = Status::Error(e.clone());
                    return Err(e);
                }
                dde.diff(
                    self.t + self.c[i] * self.h,
                    &y_stage,
                    &y_delayed,
                    &mut self.k[i],
                );
            }
            evals.function += self.stages - 1;

            // Combine stages
            let mut y_next = self.y;
            for i in 0..self.stages {
                y_next += self.k[i] * (self.b[i] * self.h);
            }

            // Convergence check (if iterating)
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
                    break;
                }
                if iter_idx == max_iter - 1 {
                    dde_iteration_failed = dde_iteration_error > T::from_f64(1e-6).unwrap();
                }
            }
            y_next_candidate_iter = y_next;

            // Derivative at t+h for current candidate
            dde.lags(self.t + self.h, &y_next_candidate_iter, &mut delays);
            if let Err(e) = self.lagvals(self.t + self.h, &delays, &mut y_delayed, phi) {
                self.status = Status::Error(e.clone());
                return Err(e);
            }
            dde.diff(
                self.t + self.h,
                &y_next_candidate_iter,
                &y_delayed,
                &mut dydt_next_candidate_iter,
            );
            evals.function += 1;
        }

        // Iteration failed: reduce h and retry
        if dde_iteration_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            if L > 0
                && min_delay_abs > T::zero()
                && self.h.abs() < T::from_f64(2.0).unwrap() * min_delay_abs
            {
                self.h = min_delay_abs * sign;
            }
            self.status = Status::RejectedStep;
            return Ok(evals);
        }

        // Store current state before update for interpolation
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Advance state
        self.t += self.h;
        self.y = y_next_candidate_iter;

        // Derivative for next step
        if self.fsal {
            self.dydt = self.k[S - 1];
        } else {
            dde.lags(self.t, &self.y, &mut delays);
            if let Err(e) = self.lagvals(self.t, &delays, &mut y_delayed, phi) {
                self.status = Status::Error(e.clone());
                return Err(e);
            }
            dde.diff(self.t, &self.y, &y_delayed, &mut self.dydt);
            evals.function += 1;
        }

        // Dense output stages
        if self.bi.is_some() {
            for i in 0..(I - S) {
                let mut y_stage_dense = self.y_prev;
                for j in 0..self.stages + i {
                    y_stage_dense += self.k[j] * (self.a[self.stages + i][j] * self.h);
                }
                let t_stage = self.t_prev + self.c[self.stages + i] * self.h;
                dde.lags(t_stage, &y_stage_dense, &mut delays);
                if let Err(e) = self.lagvals(t_stage, &delays, &mut y_delayed, phi) {
                    self.status = Status::Error(e.clone());
                    return Err(e);
                }
                dde.diff(
                    self.t_prev + self.c[self.stages + i] * self.h,
                    &y_stage_dense,
                    &y_delayed,
                    &mut self.k[self.stages + i],
                );
            }
            evals.function += I - S;
        }

        // Append to history and prune
        self.history.push_back((self.t, self.y, self.dydt));
        if let Some(max_delay) = self.max_delay {
            let cutoff_time = self.t - max_delay;
            while let Some((t_front, _, _)) = self.history.get(1) {
                if *t_front < cutoff_time {
                    self.history.pop_front();
                } else {
                    break;
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
        delays: &[T; L],
        y_delayed: &mut [Y; L],
        phi: &H,
    ) -> Result<(), Error<T, Y>>
    where
        H: Fn(T) -> Y,
    {
        for i in 0..L {
            let t_delayed = t_stage - delays[i];

            // Check if delayed time falls within the history period (t_delayed <= t0)
            if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                y_delayed[i] = phi(t_delayed);
            // If t_delayed is after t_prev then use interpolation function
            } else if (t_delayed - self.t_prev) * self.h.signum() > T::default_epsilon() {
                if self.bi.is_some() {
                    let s = (t_delayed - self.t_prev) / self.h_prev;

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
                            y_interp += self.k[i] * (cont[i] * self.h_prev);
                        }
                    }
                    y_delayed[i] = y_interp;
                } else {
                    y_delayed[i] = cubic_hermite_interpolate(
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
                            y_delayed[i] = cubic_hermite_interpolate(
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
                    return Err(Error::InsufficientHistory {
                        t_delayed,
                        t_prev: self.t_prev,
                        t_curr: self.t,
                    });
                }
            }
        }
        Ok(())
    }
}

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Interpolation<T, Y> for ExplicitRungeKutta<Delay, Fixed, T, Y, D, O, S, I>
{
    /// Interpolates the solution at time `t_interp` within the last accepted step.
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        let dir = self.h.signum();
        if (t_interp - self.t_prev) * dir < T::zero() || (t_interp - self.t) * dir > T::zero() {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // If method has dense output coefficients, use them
        if self.bi.is_some() {
            let s = (t_interp - self.t_prev) / self.h_prev;

            let bi = self.bi.as_ref().unwrap();

            let mut cont = [T::zero(); I];
            for i in 0..self.dense_stages {
                cont[i] = bi[i][self.order - 1];
                for j in (0..self.order - 1).rev() {
                    cont[i] = cont[i] * s + bi[i][j];
                }
                cont[i] *= s;
            }

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
