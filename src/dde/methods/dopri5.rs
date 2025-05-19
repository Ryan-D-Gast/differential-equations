//! Dormand-Prince 5(4) NumericalMethod for Delay Differential Equations.

use crate::{
    Error, Status,
    alias::Evals,
    dde::{DDE, NumericalMethod, methods::h_init::h_init},
    interpolate::Interpolation,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};
use std::collections::VecDeque;

/// Dormand-Prince 5(4) method adapted for DDEs.
/// 5th order method with embedded 4th order error estimation and
/// 5th order dense output. FSAL property.
/// 7 stages, 6 function evaluations per step.
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
/// use differential_equations::dde::methods::DOPRI5;
/// use nalgebra::{Vector2, vector};
///
/// let mut dopri5 = DOPRI5::new()
///    .rtol(1e-6)
///    .atol(1e-6);
///
/// let t0 = 0.0;
/// let tf = 5.0;
/// let y0 = vector![1.0, 0.0];
/// let phi = |t| { // History function for t <= t0
///     if t <= 0.0 { vector![1.0, 0.0] } else { panic!("phi called for t > t0") }
/// };
/// struct ExampleDDE;
/// impl DDE<1, f64, Vector2<f64>> for ExampleDDE {
///     fn diff(&self, t: f64, y: &Vector2<f64>, yd: &[Vector2<f64>; 1], dydt: &mut Vector2<f64>) {
///        dydt[0] = yd[0][1];
///        dydt[1] = -yd[0][0] - 1.0 * y[1];
///     }
///
///     fn lags(&self, t: f64, y: &Vector2<f64>, lags: &mut [f64; 1]) {
///         lags[0] = 1.0; // Constant delay tau = 1.0
///     }
/// }
/// let problem = DDEProblem::new(ExampleDDE, t0, tf, y0, phi);
/// let solution = problem.solve(&mut dopri5).unwrap();
///
/// let (t, y) = solution.last().unwrap();
/// println!("DOPRI5 Solution at t={}: ({}, {})", t, y[0], y[1]);
/// ```
///
/// # Settings
/// * `rtol`, `atol`, `h0`, `h_max`, `h_min`, `max_steps`, `safe`, `fac1`, `fac2`, `beta`, `max_delay`.
///
/// # Default Settings
/// * `rtol`   - 1e-3
/// * `atol`   - 1e-6
/// * `h0`     - None
/// * `h_max`   - None
/// * `h_min`   - 0.0
/// * `max_steps` - 100_000
/// * `safe`   - 0.9
/// * `fac1`   - 0.2
/// * `fac2`   - 10.0
/// * `beta`   - 0.04 (PI stabilization enabled by default for DOPRI5)
/// * `max_delay` - None
pub struct DOPRI5<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> {
    pub h0: T,
    t: T,
    y: V,
    h: T,
    pub rtol: T,
    pub atol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub safe: T,
    pub fac1: T,
    pub fac2: T,
    pub beta: T,
    pub max_delay: Option<T>,
    expo1: T,
    facc1: T,
    facc2: T,
    facold: T,
    fac11: T,
    fac: T,
    status: Status<T, V, D>,
    steps: usize,
    n_accepted: usize,
    a: [[T; 7]; 7], // Butcher A matrix, 7x7
    b: [T; 7],      // Coefficients for 5th order solution
    c: [T; 7],      // Time points for stages
    er: [T; 7],     // Error estimation coefficients
    d: [T; 7],      // Dense output coefficients
    k: [V; 7],      // k[0] is f(t,y) at step start, k[1-5] are stages, k[6] is f(t+h, ysti)
    y_old: V,
    t_old: T,
    h_old: T,
    cont: [V; 5],                             // For 5-coefficient dense output
    cont_buffer: VecDeque<(T, T, T, [V; 5])>, // Store 5-coeff cont
    phi: Option<H>,
    t0: T,
    tf: T,
    posneg: T,
    lags: [T; L],
    yd: [V; L],
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData>
    NumericalMethod<L, T, V, H, D> for DOPRI5<L, T, V, H, D>
{
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &V, phi: H) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>,
    {
        let mut evals = Evals::new();

        self.t = t0;
        self.y = *y0;
        self.t0 = t0;
        self.tf = tf;
        self.posneg = (tf - t0).signum();
        self.phi = Some(phi);

        if L > 0 {
            dde.lags(self.t, &self.y, &mut self.lags);
            for i in 0..L {
                if self.lags[i] <= T::zero() {
                    return Err(Error::BadInput {
                        msg: "All lags must be positive.".to_string(),
                    });
                }
                let t_delayed = self.t - self.lags[i];
                if (t_delayed - self.t0) * self.posneg > T::default_epsilon() {
                    return Err(Error::BadInput {
                        msg: format!(
                            "Initial delayed time {} is out of history range (t <= {}).",
                            t_delayed, self.t0
                        ),
                    });
                }
                self.yd[i] = (self.phi.as_ref().unwrap())(t_delayed);
            }
        }
        dde.diff(self.t, &self.y, &self.yd, &mut self.k[0]); // k1 = f(t0, y0, yd(t0-tau))
        evals.fcn += 1;

        if self.h0 == T::zero() {
            let h_est = h_init(
                dde,
                self.t,
                self.tf,
                &self.y,
                5,
                self.rtol,
                self.atol,
                self.h_min,
                self.h_max,
                self.phi.as_ref().unwrap(),
                &self.k[0],
                &mut evals,
            );
            self.h0 = h_est;
        }

        match validate_step_size_parameters::<T, V, D>(
            self.h0, self.h_min, self.h_max, self.t, self.tf,
        ) {
            Ok(h0_validated) => self.h = h0_validated,
            Err(status) => return Err(status),
        }

        self.t_old = self.t;
        self.y_old = self.y;
        self.h_old = self.h;

        self.steps = 0;
        self.n_accepted = 0;
        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<F>(&mut self, dde: &F) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>,
    {
        let mut evals = Evals::new();

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

        let t_current_step_start = self.t;
        let y_current_step_start = self.y;
        let k0_at_step_start = self.k[0]; // f(t,y) from previous step or init

        let mut min_lag_abs = T::infinity();
        if L > 0 {
            // Estimate y at t+h using Euler step for lag calculation
            let temp_y_for_lags = y_current_step_start + k0_at_step_start * self.h; // Simplified Euler for lag estimate
            dde.lags(
                t_current_step_start + self.h,
                &temp_y_for_lags,
                &mut self.lags,
            );
            for i in 0..L {
                min_lag_abs = min_lag_abs.min(self.lags[i].abs());
            }
        }

        let max_iter: usize = if L > 0 && min_lag_abs < self.h.abs() && min_lag_abs > T::zero() {
            5
        } else {
            1
        };

        let mut y_new_from_iter = y_current_step_start; // Will be updated in the loop
        let mut k_fnew_iter = V::zeros(); // f(t+h, y_new_from_iter)
        let mut k_stages_iter = [V::zeros(); 7]; // Stores k[1] to k[6] for the current iteration

        let mut y_for_errit_prev_iter = y_current_step_start;
        let mut iteration_failed_to_converge = false;

        for iter_idx in 0..max_iter {
            if iter_idx > 0 {
                y_for_errit_prev_iter = y_new_from_iter; // y_new from previous iteration
            }

            // Calculate k_stages_iter[1] to k_stages_iter[5] (stages 2 to 6 of RK method)
            // These use k0_at_step_start and y_current_step_start for the `y` part of RK sum.
            for j in 1..=5 {
                // Loop for stages k_stages_iter[1] to k_stages_iter[5]
                let mut yi_stage_sum = k0_at_step_start * self.a[j][0];
                for l_idx in 1..j {
                    // k_stages_iter[1]...k_stages_iter[j-1]
                    yi_stage_sum += k_stages_iter[l_idx] * self.a[j][l_idx];
                }
                let yi = y_current_step_start + yi_stage_sum * self.h;
                let ti = t_current_step_start + self.c[j] * self.h;

                if L > 0 {
                    dde.lags(ti, &yi, &mut self.lags);
                    self.lagvals(ti, &yi);
                }
                dde.diff(ti, &yi, &self.yd, &mut k_stages_iter[j]);
            }
            evals.fcn += 5;

            // Calculate k_stages_iter[6] (stage 7, using ysti)
            let mut ysti_sum = k0_at_step_start * self.a[6][0]; // a[6][1] is 0 for DOPRI5
            for l_idx in 2..=5 {
                // k_stages_iter[2]...k_stages_iter[5]
                ysti_sum += k_stages_iter[l_idx] * self.a[6][l_idx];
            }
            let ysti = y_current_step_start + ysti_sum * self.h;
            let t_sti = t_current_step_start + self.c[6] * self.h; // c[6] is 1.0

            if L > 0 {
                dde.lags(t_sti, &ysti, &mut self.lags);
                self.lagvals(t_sti, &ysti);
            }
            dde.diff(t_sti, &ysti, &self.yd, &mut k_stages_iter[6]);
            evals.fcn += 1;

            // Calculate y_new_from_iter (5th order solution)
            let mut sum_for_y_new = k0_at_step_start * self.b[0]; // b[1] is 0 for DOPRI5
            for l_idx in 2..=6 {
                // k_stages_iter[2]...k_stages_iter[6]
                sum_for_y_new += k_stages_iter[l_idx] * self.b[l_idx];
            }
            y_new_from_iter = y_current_step_start + sum_for_y_new * self.h;

            // Calculate f(t+h, y_new_from_iter) for FSAL and dense output
            let t_new_val_for_k_fnew = t_current_step_start + self.h; // c[6] is 1.0, effectively t+h
            if L > 0 {
                dde.lags(t_new_val_for_k_fnew, &y_new_from_iter, &mut self.lags);
                self.lagvals(t_new_val_for_k_fnew, &y_new_from_iter);
            }
            dde.diff(
                t_new_val_for_k_fnew,
                &y_new_from_iter,
                &self.yd,
                &mut k_fnew_iter,
            );
            evals.fcn += 1;

            if max_iter > 1 && iter_idx > 0 {
                let mut errit_val = T::zero();
                let n_dim = y_current_step_start.len();
                for i_dim in 0..n_dim {
                    let scale = self.atol
                        + self.rtol
                            * y_for_errit_prev_iter
                                .get(i_dim)
                                .abs()
                                .max(y_new_from_iter.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val =
                            y_new_from_iter.get(i_dim) - y_for_errit_prev_iter.get(i_dim);
                        errit_val += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    errit_val = (errit_val / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if errit_val <= self.rtol * T::from_f64(0.1).unwrap() {
                    break;
                }
                if iter_idx == max_iter - 1 {
                    // Check on the last iteration
                    iteration_failed_to_converge =
                        errit_val > self.rtol * T::from_f64(0.1).unwrap();
                }
            }
        }

        if iteration_failed_to_converge {
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * self.posneg;
            if L > 0
                && min_lag_abs > T::zero()
                && self.h.abs() < T::from_f64(2.0).unwrap() * min_lag_abs
            {
                self.h = min_lag_abs * self.posneg; // Avoid stepping too far into a discontinuity
            }
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
            // Restore k[0] as it might have been used if we had a k_stages_iter[0]
            self.k[0] = k0_at_step_start;
            return Ok(evals);
        }

        let mut err_final = T::zero();
        let n = y_current_step_start.len();
        for i in 0..n {
            let sk = self.atol
                + self.rtol
                    * y_current_step_start
                        .get(i)
                        .abs()
                        .max(y_new_from_iter.get(i).abs());
            // Error estimation using k0_at_step_start and k_stages_iter[1] to k_stages_iter[6]
            // er[1] is 0 for DOPRI5
            let mut err_comp_sum = k0_at_step_start.get(i) * self.er[0];
            for j in 2..=6 {
                // k_stages_iter[2] to k_stages_iter[6]
                err_comp_sum += k_stages_iter[j].get(i) * self.er[j];
            }
            let erri = self.h * err_comp_sum;
            if sk > T::zero() {
                err_final += (erri / sk).powi(2);
            }
        }
        if n > 0 {
            err_final = (err_final / T::from_usize(n).unwrap()).sqrt();
        }

        self.fac11 = err_final.powf(self.expo1);
        let fac_beta = if self.beta > T::zero() && self.facold > T::zero() {
            self.facold.powf(self.beta)
        } else {
            T::one()
        };
        self.fac = self.fac11 / fac_beta;
        self.fac = self.facc2.max(self.facc1.min(self.fac / self.safe));
        let mut h_new_final = self.h / self.fac;

        let t_new_val = t_current_step_start + self.h;

        if err_final <= T::one() {
            self.facold = err_final.max(T::from_f64(1.0e-4).unwrap());
            self.n_accepted += 1;

            // Dense output coefficients
            let ydiff = y_new_from_iter - y_current_step_start;
            let bspl = k0_at_step_start * self.h - ydiff;

            self.cont[0] = y_current_step_start;
            self.cont[1] = ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - k_fnew_iter * self.h - bspl;

            // d[1] is 0 for DOPRI5
            let mut d_sum = k0_at_step_start * self.d[0];
            for j in 2..=6 {
                // k_stages_iter[2] to k_stages_iter[6]
                d_sum += k_stages_iter[j] * self.d[j];
            }
            self.cont[4] = d_sum * self.h;

            self.cont_buffer
                .push_back((t_current_step_start, t_new_val, self.h, self.cont));

            if let Some(max_delay_val) = self.max_delay {
                let prune_time = if self.posneg > T::zero() {
                    t_new_val - max_delay_val
                } else {
                    t_new_val + max_delay_val
                };
                while let Some((buf_t_start, buf_t_end, _, _)) = self.cont_buffer.front() {
                    if (self.posneg > T::zero() && *buf_t_end < prune_time)
                        || (self.posneg < T::zero() && *buf_t_start > prune_time)
                    {
                        self.cont_buffer.pop_front();
                    } else {
                        break;
                    }
                }
            }

            self.y_old = y_current_step_start;
            self.t_old = t_current_step_start;
            self.h_old = self.h;

            self.k[0] = k_fnew_iter; // FSAL: f(t+h, y_new) becomes k1 for next step
            self.y = y_new_from_iter;
            self.t = t_new_val;

            if let Status::RejectedStep = self.status {
                h_new_final = self.h_old.min(h_new_final);
                self.status = Status::Solving;
            }
        } else {
            h_new_final = self.h / self.facc1.min(self.fac11 / self.safe);
            self.status = Status::RejectedStep;
        }

        self.steps += 1;
        self.h = constrain_step_size(h_new_final, self.h_min, self.h_max);
        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }
    fn y(&self) -> &V {
        &self.y
    }
    fn t_prev(&self) -> T {
        self.t_old
    }
    fn y_prev(&self) -> &V {
        &self.y_old
    }
    fn h(&self) -> T {
        self.h
    }
    fn set_h(&mut self, h: T) {
        self.h = h;
    }
    fn status(&self) -> &Status<T, V, D> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, V, D>) {
        self.status = status;
    }
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> Interpolation<T, V>
    for DOPRI5<L, T, V, H, D>
{
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        if ((t_interp - self.t_old) * self.posneg < T::zero()
            || (t_interp - self.t) * self.posneg > T::zero())
            && (t_interp - self.t_old).abs() > T::default_epsilon()
            && (t_interp - self.t).abs() > T::default_epsilon()
        {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_old,
                t_curr: self.t,
            });
        }

        let s = if self.h_old == T::zero() {
            if (t_interp - self.t_old).abs() < T::default_epsilon() {
                T::zero()
            } else {
                T::one()
            }
        } else {
            (t_interp - self.t_old) / self.h_old
        };
        let s1 = T::one() - s;

        // Use the 5-coefficient dense output formula
        let y_interp = self.cont[0]
            + (self.cont[1] + (self.cont[2] + (self.cont[3] + self.cont[4] * s1) * s) * s1) * s;
        Ok(y_interp)
    }
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> DOPRI5<L, T, V, H, D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }
    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }
    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }
    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }
    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    pub fn safe(mut self, safe: T) -> Self {
        self.safe = safe;
        self
    }
    pub fn fac1(mut self, fac1: T) -> Self {
        self.fac1 = fac1;
        self.facc1 = T::one() / fac1;
        self
    }
    pub fn fac2(mut self, fac2: T) -> Self {
        self.fac2 = fac2;
        self.facc2 = T::one() / fac2;
        self
    }
    pub fn beta(mut self, beta: T) -> Self {
        self.beta = beta;
        self
    }
    pub fn max_delay(mut self, max_delay: T) -> Self {
        self.max_delay = Some(max_delay.abs());
        self
    }

    fn lagvals(&mut self, t_stage: T, _y_stage: &V) {
        for i in 0..L {
            let t_delayed = t_stage - self.lags[i];
            if (t_delayed - self.t0) * self.posneg <= T::default_epsilon() {
                self.yd[i] = (self.phi.as_ref().unwrap())(t_delayed);
            } else {
                let mut found_in_buffer = false;
                for (buf_t_start, buf_t_end, buf_h, buf_cont) in self.cont_buffer.iter().rev() {
                    if (t_delayed - *buf_t_start) * self.posneg >= -T::default_epsilon()
                        && (t_delayed - *buf_t_end) * self.posneg <= T::default_epsilon()
                    {
                        let s = if *buf_h == T::zero() {
                            if (t_delayed - *buf_t_start).abs() < T::default_epsilon() {
                                T::zero()
                            } else {
                                T::one()
                            }
                        } else {
                            (t_delayed - *buf_t_start) / *buf_h
                        };
                        self.yd[i] = buf_cont[0]
                            + (buf_cont[1]
                                + (buf_cont[2] + (buf_cont[3] + buf_cont[4] * (T::one() - s)) * s)
                                    * (T::one() - s))
                                * s;
                        found_in_buffer = true;
                        break;
                    }
                }
                if !found_in_buffer {
                    if let Some((buf_t_start, _buf_t_end, buf_h, buf_cont)) =
                        self.cont_buffer.back()
                    {
                        let s = if *buf_h == T::zero() {
                            T::one()
                        } else {
                            (t_delayed - *buf_t_start) / *buf_h
                        };
                        self.yd[i] = buf_cont[0]
                            + (buf_cont[1]
                                + (buf_cont[2] + (buf_cont[3] + buf_cont[4] * (T::one() - s)) * s)
                                    * (T::one() - s))
                                * s;
                    } else {
                        self.yd[i] = (self.phi.as_ref().unwrap())(t_delayed);
                    }
                }
            }
        }
    }
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> Default
    for DOPRI5<L, T, V, H, D>
{
    fn default() -> Self {
        // Convert coefficient arrays from f64 to type T
        let a_conv = DOPRI5_A.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b_conv = DOPRI5_B.map(|x| T::from_f64(x).unwrap());
        let c_conv = DOPRI5_C.map(|x| T::from_f64(x).unwrap());
        let er_conv = DOPRI5_E.map(|x| T::from_f64(x).unwrap());
        let d_conv = DOPRI5_D.map(|x| T::from_f64(x).unwrap());

        let expo1_final = T::one() / T::from_f64(5.0).unwrap();

        let fac1_default = T::from_f64(0.2).unwrap();
        let fac2_default = T::from_f64(10.0).unwrap();
        let beta_default = T::from_f64(0.04).unwrap();

        DOPRI5 {
            t: T::zero(),
            y: V::zeros(),
            h: T::zero(),
            h0: T::zero(),
            rtol: T::from_f64(1e-3).unwrap(),
            atol: T::from_f64(1e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 100_000,
            safe: T::from_f64(0.9).unwrap(),
            fac1: fac1_default,
            fac2: fac2_default,
            beta: beta_default,
            max_delay: None,
            expo1: expo1_final,
            facc1: T::one() / fac1_default,
            facc2: T::one() / fac2_default,
            facold: T::from_f64(1.0e-4).unwrap(),
            fac11: T::zero(),
            fac: T::zero(),
            status: Status::Uninitialized,
            steps: 0,
            n_accepted: 0,
            a: a_conv,
            b: b_conv,
            c: c_conv,
            er: er_conv,
            d: d_conv,
            k: [V::zeros(); 7],
            y_old: V::zeros(),
            t_old: T::zero(),
            h_old: T::zero(),
            cont: [V::zeros(); 5], // Initialize 5-element cont
            cont_buffer: VecDeque::new(),
            phi: None,
            t0: T::zero(),
            tf: T::zero(),
            posneg: T::zero(),
            lags: [T::zero(); L],
            yd: [V::zeros(); L],
        }
    }
}

// DOPRI5 Butcher Tableau

// A matrix (7x7, lower triangular)
const DOPRI5_A: [[f64; 7]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
        0.0,
    ],
    [
        // Coefficients for ysti (used to compute k[6])
        35.0 / 384.0,
        0.0, // a[6][1] is 0
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
];

// C coefficients (nodes)
const DOPRI5_C: [f64; 7] = [
    0.0,       // C1 (for k[0])
    0.2,       // C2 (for k[1])
    0.3,       // C3 (for k[2])
    0.8,       // C4 (for k[3])
    8.0 / 9.0, // C5 (for k[4])
    1.0,       // C6 (for k[5])
    1.0,       // C7 (for k[6], t_new)
];

// B coefficients (weights for main method - 5th order)
const DOPRI5_B: [f64; 7] = [
    35.0 / 384.0,     // B1 (for k[0])
    0.0,              // B2 (for k[1])
    500.0 / 1113.0,   // B3 (for k[2])
    125.0 / 192.0,    // B4 (for k[3])
    -2187.0 / 6784.0, // B5 (for k[4])
    11.0 / 84.0,      // B6 (for k[5])
    0.0,              // B7 (for k[6])
];

// Error estimation coefficients (5th order - 4th order)
const DOPRI5_E: [f64; 7] = [
    71.0 / 57600.0,      // E1 (for k[0])
    0.0,                 // E2 (for k[1])
    -71.0 / 16695.0,     // E3 (for k[2])
    71.0 / 1920.0,       // E4 (for k[3])
    -17253.0 / 339200.0, // E5 (for k[4])
    22.0 / 525.0,        // E6 (for k[5])
    -1.0 / 40.0,         // E7 (for k[6])
];

// Dense output coefficients
const DOPRI5_D: [f64; 7] = [
    -12715105075.0 / 11282082432.0,  // D1 (for k[0])
    0.0,                             // D2 (for k[1])
    87487479700.0 / 32700410799.0,   // D3 (for k[2])
    -10690763975.0 / 1880347072.0,   // D4 (for k[3])
    701980252875.0 / 199316789632.0, // D5 (for k[4])
    -1453857185.0 / 822651844.0,     // D6 (for k[5])
    69997945.0 / 29380423.0,         // D7 (for k[6])
];
