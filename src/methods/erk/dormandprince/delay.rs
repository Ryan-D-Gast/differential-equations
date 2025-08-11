//! Dormand–Prince explicit Runge–Kutta methods for Delay Differential Equations (DDEs)

use crate::{
    dde::{DDE, DelayNumericalMethod},
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    methods::{Delay, DormandPrince, ExplicitRungeKutta, h_init::InitialStepSize},
    stats::Evals,
    status::Status,
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
> DelayNumericalMethod<L, T, Y, H, D>
    for ExplicitRungeKutta<Delay, DormandPrince, T, Y, D, O, S, I>
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

        // Initialize solver state
        self.t0 = t0;
        self.t = t0;
        self.y = *y0;
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.status = Status::Initialized;
        self.steps = 0;
        self.stiffness_counter = 0;
        self.non_stiffness_counter = 0;
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
                    msg: format!("Delayed time {} is beyond initial time {}", t_delayed, t0),
                });
            }
            y_delayed[i] = phi(t_delayed);
        }

        // Initial derivative
        dde.diff(self.t, &self.y, &y_delayed, &mut self.k[0]);
        self.dydt = self.k[0];
        evals.function += 1;
        self.dydt_prev = self.dydt;

        // Seed history
        self.history.push_back((self.t, self.y, self.dydt));

        // Initial step size
        if self.h0 == T::zero() {
            self.h0 = InitialStepSize::<Delay>::compute(
                dde, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max, phi,
                &self.k[0], &mut evals,
            );
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

        // Decide if delay iteration is needed
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
        let mut y_next_est_prev = self.y;
        let mut dde_iter_failed = false;
        let mut err_norm: T = T::zero();
        let mut y_last_stage = Y::zeros();

        // DDE iteration loop
        for it in 0..max_iter {
            if it > 0 {
                y_next_est_prev = y_next_est;
            }

            // Compute stages
            let mut y_stage = Y::zeros();
            for i in 1..self.stages {
                y_stage = Y::zeros();
                for j in 0..i {
                    y_stage += self.k[j] * self.a[i][j];
                }
                y_stage = self.y + y_stage * self.h;

                // Delayed states for this stage
                dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut delays);
                if let Err(e) =
                    self.lagvals(self.t + self.c[i] * self.h, &delays, &mut y_delayed, phi)
                {
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

            // Keep last stage for stiffness detection
            y_last_stage = y_stage;

            // RK combination
            let mut yseg = Y::zeros();
            for i in 0..self.stages {
                yseg += self.k[i] * self.b[i];
            }

            let y_new = self.y + yseg * self.h;

            // Dormand–Prince error estimation
            let er = self.er.unwrap();
            let n = self.y.len();
            let mut err_val = T::zero();
            let mut err2 = T::zero();
            let mut erri;
            for i in 0..n {
                let sk = self.atol + self.rtol * self.y.get(i).abs().max(y_new.get(i).abs());
                erri = T::zero();
                for j in 0..self.stages {
                    erri += er[j] * self.k[j].get(i);
                }
                err_val += (erri / sk).powi(2);
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
            err_norm =
                self.h.abs() * err_val * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();

            // Convergence check (if iterating)
            if max_iter > 1 && it > 0 {
                let mut dde_iteration_error = T::zero();
                let n_dim = self.y.len();
                for i_dim in 0..n_dim {
                    let scale = self.atol
                        + self.rtol * y_next_est_prev.get(i_dim).abs().max(y_new.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val = y_new.get(i_dim) - y_next_est_prev.get(i_dim);
                        dde_iteration_error += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    dde_iteration_error =
                        (dde_iteration_error / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if dde_iteration_error <= self.rtol * T::from_f64(0.1).unwrap() {
                    break;
                }
                if it == max_iter - 1 {
                    dde_iter_failed = dde_iteration_error > self.rtol * T::from_f64(0.1).unwrap();
                }
            }
            y_next_est = y_new;
        }

        // Iteration failed: reduce h and retry
        if dde_iter_failed {
            let sign = self.h.signum();
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * sign;
            if L > 0
                && min_delay_abs > T::zero()
                && self.h.abs() < T::from_f64(2.0).unwrap() * min_delay_abs
            {
                self.h = min_delay_abs * sign;
            }
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
            return Ok(evals);
        }

        // Step size control
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err_norm.powf(-error_exponent);

        // Clamp scale factor
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Accept/reject
        if err_norm <= T::one() {
            let y_new = y_next_est;
            let t_new = self.t + self.h;

            // Derivative at new point
            dde.lags(t_new, &y_new, &mut delays);
            if let Err(e) = self.lagvals(t_new, &delays, &mut y_delayed, phi) {
                self.status = Status::Error(e.clone());
                return Err(e);
            }
            dde.diff(t_new, &y_new, &y_delayed, &mut self.dydt);
            evals.function += 1;
            // Stiffness detection (every 100 steps)
            let n_stiff_threshold = 100;
            if self.steps % n_stiff_threshold == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = {
                    let mut yseg = Y::zeros();
                    for i in 0..self.stages {
                        yseg += self.k[i] * self.b[i];
                    }
                    yseg - self.k[S - 1]
                };
                for i in 0..sqr.len() {
                    stdnum += sqr.get(i).powi(2);
                }
                let sqr = self.dydt - y_last_stage;
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

            // Prepare dense output / interpolation
            self.cont[0] = self.y;
            let ydiff = y_new - self.y;
            self.cont[1] = ydiff;
            let bspl = self.k[0] * self.h - ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.dydt * self.h - bspl;

            // Dense output stages
            if let Some(bi) = &self.bi {
                if I > S {
                    self.k[self.stages] = self.dydt;
                    for i in S + 1..I {
                        let mut y_stage = Y::zeros();
                        for j in 0..i {
                            y_stage += self.k[j] * self.a[i][j];
                        }
                        y_stage = self.y + y_stage * self.h;

                        dde.lags(self.t + self.c[i] * self.h, &y_stage, &mut delays);
                        for lag_idx in 0..L {
                            let t_delayed = (self.t + self.c[i] * self.h) - delays[lag_idx];

                            if (t_delayed - self.t0) * self.h.signum() <= T::default_epsilon() {
                                y_delayed[lag_idx] = phi(t_delayed);
                            } else if (t_delayed - self.t_prev) * self.h.signum()
                                > T::default_epsilon()
                            {
                                if self.bi.is_some() {
                                    let theta = (t_delayed - self.t_prev) / self.h_prev;
                                    let one_minus_theta = T::one() - theta;
                                    let ilast = self.cont.len() - 1;
                                    let poly =
                                        (1..ilast).rev().fold(self.cont[ilast], |acc, cont_i| {
                                            let factor = if cont_i >= 4 {
                                                if (ilast - cont_i) % 2 == 1 {
                                                    one_minus_theta
                                                } else {
                                                    theta
                                                }
                                            } else if cont_i % 2 == 1 {
                                                one_minus_theta
                                            } else {
                                                theta
                                            };
                                            acc * factor + self.cont[cont_i]
                                        });
                                    y_delayed[lag_idx] = self.cont[0] + poly * theta;
                                } else {
                                    y_delayed[lag_idx] = cubic_hermite_interpolate(
                                        self.t_prev,
                                        self.t,
                                        &self.y_prev,
                                        &self.y,
                                        &self.dydt_prev,
                                        &self.dydt,
                                        t_delayed,
                                    );
                                }
                            } else {
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
                                            y_delayed[lag_idx] = cubic_hermite_interpolate(
                                                *t_left, *t_right, y_left, y_right, dydt_left,
                                                dydt_right, t_delayed,
                                            );
                                            found_interpolation = true;
                                            break;
                                        }
                                        prev_entry = curr_entry;
                                    }
                                }
                                if !found_interpolation {
                                    return Err(Error::InsufficientHistory {
                                        t_delayed,
                                        t_prev: self.t_prev,
                                        t_curr: self.t,
                                    });
                                }
                            }
                        }
                        dde.diff(
                            self.t + self.c[i] * self.h,
                            &y_stage,
                            &y_delayed,
                            &mut self.k[i],
                        );
                        evals.function += 1;
                    }
                }

                // Dense output coefficients
                for i in 4..self.order {
                    self.cont[i] = Y::zeros();
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

            // Advance state
            self.t = t_new;
            self.y = y_new;
            self.k[0] = self.dydt;

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
            if let Status::RejectedStep = self.status {
                self.status = Status::Solving;
                scale = scale.min(T::one());
            }
        } else {
            // Step rejected
            self.status = Status::RejectedStep;
        }

        // Update step size
        self.h *= scale;
        // Enforce bounds
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
    ExplicitRungeKutta<Delay, DormandPrince, T, Y, D, O, S, I>
{
    fn lagvals<const L: usize, H>(
        &mut self,
        t_stage: T,
        lags: &[T; L],
        yd: &mut [Y; L],
        phi: &H,
    ) -> Result<(), Error<T, Y>>
    where
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
                    let theta = (t_delayed - self.t_prev) / self.h_prev;
                    let one_minus_theta = T::one() - theta;

                    // Functional implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
                    let ilast = self.cont.len() - 1;
                    let poly = (1..ilast).rev().fold(self.cont[ilast], |acc, i| {
                        let factor = if i >= 4 {
                            if (ilast - i) % 2 == 1 {
                                one_minus_theta
                            } else {
                                theta
                            }
                        } else if i % 2 == 1 {
                            one_minus_theta
                        } else {
                            theta
                        };
                        acc * factor + self.cont[i]
                    });

                    // Final multiplication by theta for the outermost level
                    let y_interp = self.cont[0] + poly * theta;
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
                                *t_left, *t_right, y_left, y_right, dydt_left, dydt_right,
                                t_delayed,
                            );
                            found_interpolation = true;
                            break;
                        }
                        prev_entry = curr_entry;
                    }
                }
                // If not found in history, this indicates insufficient history in buffer
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
    Interpolation<T, Y> for ExplicitRungeKutta<Delay, DormandPrince, T, Y, D, O, S, I>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Check if interpolation is out of bounds
        let dir = (self.t - self.t_prev).signum();
        if (t_interp - self.t_prev) * dir < T::zero() || (t_interp - self.t) * dir > T::zero() {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // Evaluate the interpolation polynomial at the requested time
        let theta = (t_interp - self.t_prev) / self.h_prev;
        let one_minus_theta = T::one() - theta;

        // Functional implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
        let ilast = self.cont.len() - 1;
        let poly = (1..ilast).rev().fold(self.cont[ilast], |acc, i| {
            let factor = if i >= 4 {
                if (ilast - i) % 2 == 1 {
                    one_minus_theta
                } else {
                    theta
                }
            } else if i % 2 == 1 {
                one_minus_theta
            } else {
                theta
            };
            acc * factor + self.cont[i]
        });

        // Final multiplication by theta for the outermost level
        let y_interp = self.cont[0] + poly * theta;

        Ok(y_interp)
    }
}
