//! Adaptive Runge-Kutta methods for Delay Differential Equations (DDEs)

use crate::{
    Error, Status,
    dde::{DDE, DelayNumericalMethod},
    interpolate::{Interpolation, cubic_hermite_interpolate},
    methods::{Adaptive, Delay, ExplicitRungeKutta, h_init::InitialStepSize},
    stats::Evals,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
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
> DelayNumericalMethod<L, T, Y, H, D> for ExplicitRungeKutta<Delay, Adaptive, T, Y, D, O, S, I>
{
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &Y, phi: &H) -> Result<Evals, Error<T, Y>>
    where
        F: DDE<L, T, Y, D>,
    {
        let mut evals = Evals::new();

        // DDE requires at least one lag
        if L <= 0 {
            return Err(Error::NoLags);
        }

        // Init solver state
        self.t0 = t0;
        self.t = t0;
        self.y = *y0;
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.status = Status::Initialized;
        self.steps = 0;
        self.stiffness_counter = 0;
        self.history = VecDeque::new();

        // Delay buffers
        let mut delays = [T::zero(); L];
        let mut y_delayed = [Y::zeros(); L];

        // Initial delays and history
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

        // Initial derivative and seed history
        dde.diff(self.t, &self.y, &y_delayed, &mut self.dydt);
        evals.function += 1;
        self.dydt_prev = self.dydt; // Store initial state in history
        self.history.push_back((self.t, self.y, self.dydt));

        // Initial step size
        if self.h0 == T::zero() {
            // Adaptive step size for DDEs
            self.h0 = InitialStepSize::<Delay>::compute(
                dde, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max, phi,
                &self.k[0], &mut evals,
            );
            evals.function += 2; // h_init performs 2 function evaluations
        }

        // Validate initial step size
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

        // Validate step size
        if self.h.abs() < self.h_prev.abs() * T::from_f64(1e-14).unwrap() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y,
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y,
            });
        }

        // Max steps
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

        // Seed k[0]
        self.k[0] = self.dydt;

        // Check if delay iteration is needed
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

        let mut y_next_est = self.y;
        let mut dydt_next_est = Y::zeros();
        let mut y_next_est_prev = self.y;
        let mut dde_iter_failed = false;
        let mut err_norm: T = T::zero();

        // DDE iteration loop
        for it in 0..max_iter {
            if it > 0 {
                y_next_est_prev = y_next_est;
            }

            // Compute stages
            for i in 1..self.stages {
                let mut y_stage = self.y;
                for j in 0..i {
                    y_stage += self.k[j] * (self.a[i][j] * self.h);
                }
                // Delayed states for this stage
                let t_stage = self.t + self.c[i] * self.h;
                dde.lags(t_stage, &y_stage, &mut delays);
                if let Err(e) = self.lagvals(t_stage, &delays, &mut y_delayed, phi) {
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

            // High/low order solutions for error
            let mut y_high = self.y;
            for i in 0..self.stages {
                y_high += self.k[i] * (self.b[i] * self.h);
            }
            let mut y_low = self.y;
            if let Some(bh) = &self.bh {
                for i in 0..self.stages {
                    y_low += self.k[i] * (bh[i] * self.h);
                }
            }
            let err_vec: Y = y_high - y_low;

            // Infinity-norm-like error scaled by atol/rtol
            err_norm = T::zero();
            for n in 0..self.y.len() {
                let tol = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
                err_norm = err_norm.max((err_vec.get(n) / tol).abs());
            }

            // Iteration convergence (if iterating)
            if max_iter > 1 && it > 0 {
                let mut iter_err = T::zero();
                let n_dim = self.y.len();
                for d in 0..n_dim {
                    let scale = self.atol
                        + self.rtol * y_next_est_prev.get(d).abs().max(y_high.get(d).abs());
                    if scale > T::zero() {
                        let diff_val = y_high.get(d) - y_next_est_prev.get(d);
                        iter_err += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    iter_err = (iter_err / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if iter_err <= self.rtol * T::from_f64(0.1).unwrap() {
                    y_next_est = y_high;
                    dde.lags(self.t + self.h, &y_next_est, &mut delays);
                    if let Err(e) = self.lagvals(self.t + self.h, &delays, &mut y_delayed, phi) {
                        self.status = Status::Error(e.clone());
                        return Err(e);
                    }
                    dde.diff(self.t + self.h, &y_next_est, &y_delayed, &mut dydt_next_est);
                    evals.function += 1;
                    break;
                }
                if it == max_iter - 1 {
                    dde_iter_failed = iter_err > self.rtol * T::from_f64(0.1).unwrap();
                }
            }

            // Update candidate
            y_next_est = y_high;

            // Derivative at t+h for candidate
            dde.lags(self.t + self.h, &y_next_est, &mut delays);
            if let Err(e) = self.lagvals(self.t + self.h, &delays, &mut y_delayed, phi) {
                self.status = Status::Error(e.clone());
                return Err(e);
            }
            dde.diff(self.t + self.h, &y_next_est, &y_delayed, &mut dydt_next_est);
            evals.function += 1;
        }

        // Iteration failed: reduce h and retry
        if dde_iter_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            if min_delay_abs > T::zero() && self.h.abs() < T::from_f64(2.0).unwrap() * min_delay_abs
            {
                self.h = min_delay_abs * sign;
            }

            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
            return Ok(evals);
        }

        // Step size scale factor
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err_norm.powf(-error_exponent);
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Accept/reject
        if err_norm <= T::one() {
            // Accept
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.dydt;
            self.h_prev = self.h;

            if let Status::RejectedStep = self.status {
                // Dampen growth after rejection
                self.stiffness_counter = 0;
                scale = scale.min(T::one());
            }
            self.status = Status::Solving;

            // Dense output stages
            if self.bi.is_some() {
                for i in 0..(I - S) {
                    let mut y_stage = self.y;
                    for j in 0..self.stages + i {
                        y_stage += self.k[j] * (self.a[self.stages + i][j] * self.h);
                    }
                    let t_stage = self.t + self.c[self.stages + i] * self.h;
                    dde.lags(t_stage, &y_stage, &mut delays);
                    if let Err(e) = self.lagvals(t_stage, &delays, &mut y_delayed, phi) {
                        self.status = Status::Error(e.clone());
                        return Err(e);
                    }
                    dde.diff(
                        self.t + self.c[self.stages + i] * self.h,
                        &y_stage,
                        &y_delayed,
                        &mut self.k[self.stages + i],
                    );
                }
                evals.function += I - S;
            }

            // Advance state
            self.t += self.h;
            self.y = y_next_est;

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
        } else {
            // Reject
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            if self.stiffness_counter >= self.max_rejects {
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

        // Update step size
        self.h *= scale;
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);

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
    ExplicitRungeKutta<Delay, Adaptive, T, Y, D, O, S, I>
{
    fn lagvals<const L: usize, H>(
        &mut self,
        t_stage: T,
        delays: &[T; L],
        y_delayed: &mut [Y; L],
        phi: &H,
    ) -> Result<(), Error<T, Y>>
    where
        H: Fn(T) -> Y,
    {
        for idx in 0..L {
            let t_delayed = t_stage - delays[idx];

            // History domain (t_delayed <= t0)
            if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                y_delayed[idx] = phi(t_delayed);
            // Within last accepted step (dense if available, else Hermite)
            } else if (t_delayed - self.t_prev) * self.h.signum() > T::default_epsilon() {
                if self.bi.is_some() {
                    let theta = (t_delayed - self.t_prev) / self.h_prev;
                    let dense_coeffs = self.bi.as_ref().unwrap();

                    let mut coeffs = [T::zero(); I];
                    for s_idx in 0..I {
                        if s_idx < self.cont.len() && s_idx < dense_coeffs.len() {
                            coeffs[s_idx] = dense_coeffs[s_idx][self.dense_stages - 1];
                            for j in (0..self.dense_stages - 1).rev() {
                                coeffs[s_idx] = coeffs[s_idx] * theta + dense_coeffs[s_idx][j];
                            }
                            coeffs[s_idx] *= theta;
                        }
                    }

                    let mut y_interp = self.y_prev;
                    for s_idx in 0..I {
                        if s_idx < self.k.len() && s_idx < self.cont.len() {
                            y_interp += self.k[s_idx] * (coeffs[s_idx] * self.h_prev);
                        }
                    }
                    y_delayed[idx] = y_interp;
                } else {
                    y_delayed[idx] = cubic_hermite_interpolate(
                        self.t_prev,
                        self.t,
                        &self.y_prev,
                        &self.y,
                        &self.dydt_prev,
                        &self.dydt,
                        t_delayed,
                    );
                }
            // Between earlier history points (internal buffer)
            } else {
                // Search history for bracketing interval
                let mut found = false;
                let buffer = &self.history;
                let mut it = buffer.iter();
                if let Some(mut left) = it.next() {
                    for right in it {
                        let (t_left, y_left, dydt_left) = left;
                        let (t_right, y_right, dydt_right) = right;

                        let in_interval = if self.h.signum() > T::zero() {
                            *t_left <= t_delayed && t_delayed <= *t_right
                        } else {
                            *t_right <= t_delayed && t_delayed <= *t_left
                        };

                        if in_interval {
                            y_delayed[idx] = cubic_hermite_interpolate(
                                *t_left, *t_right, y_left, y_right, dydt_left, dydt_right,
                                t_delayed,
                            );
                            found = true;
                            break;
                        }
                        left = right;
                    }
                }
                if !found {
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
    Interpolation<T, Y> for ExplicitRungeKutta<Delay, Adaptive, T, Y, D, O, S, I>
{
    /// Interpolates the solution at a given time `t_interp`.
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        let dir = (self.t - self.t_prev).signum();
        if (t_interp - self.t_prev) * dir < T::zero() || (t_interp - self.t) * dir > T::zero() {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // If method has dense output coefficients, use them
        if self.bi.is_some() {
            // Calculate the normalized distance within the step [0, 1]
            let theta = (t_interp - self.t_prev) / self.h_prev;

            // Get the interpolation coefficients
            let dense_coeffs = self.bi.as_ref().unwrap();

            let mut coeffs = [T::zero(); I];
            // Compute the interpolation coefficients using Horner's method
            for i in 0..self.dense_stages {
                // Start with the highest-order term
                coeffs[i] = dense_coeffs[i][self.order - 1];

                // Apply Horner's method
                for j in (0..self.order - 1).rev() {
                    coeffs[i] = coeffs[i] * theta + dense_coeffs[i][j];
                }

                // Multiply by s
                coeffs[i] *= theta;
            }

            // Compute the interpolated value
            let mut y_interp = self.y_prev;
            for i in 0..I {
                y_interp += self.k[i] * coeffs[i] * self.h_prev;
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
