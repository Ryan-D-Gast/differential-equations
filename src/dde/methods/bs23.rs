//! BS23 DDENumericalMethod for Delay Differential Equations.

use crate::{
    Error, Status,
    alias::Evals,
    dde::{DDE, DDENumericalMethod, methods::h_init::h_init},
    interpolate::Interpolation,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};
use std::collections::VecDeque;

/// Bogacki-Shampine 3(2) method adapted for DDEs.
/// 3rd order method with embedded 2nd order error estimation and
/// 3rd order dense output (cubic Hermite interpolation). FSAL property.
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
/// use differential_equations::dde::methods::BS23;
/// use nalgebra::{Vector2, vector};
///
/// let mut bs23 = BS23::new()
///    .rtol(1e-5)
///    .atol(1e-5);
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
/// let solution = problem.solve(&mut bs23).unwrap();
///
/// let (t, y) = solution.last().unwrap();
/// println!("BS23 Solution at t={}: ({}, {})", t, y[0], y[1]);
/// ```
///
/// # Settings (similar to BS23)
/// * `rtol`, `atol`, `h0`, `h_max`, `h_min`, `max_steps`, `safe`, `fac1`, `fac2`, `beta`, `max_delay`.
///
/// # Default Settings (typical for BS23)
/// * `rtol`   - 1e-3
/// * `atol`   - 1e-6
/// * `h0`     - None
/// * `h_max`   - None
/// * `h_min`   - 0.0
/// * `max_steps` - 100_000
/// * `safe`   - 0.9
/// * `fac1`   - 0.2 (1/5 for order 3, can be tuned)
/// * `fac2`   - 10.0
/// * `beta`   - 0.0 (No PI stabilization by default for BS23, can be enabled)
/// * `max_delay` - None
pub struct BS23<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> {
    pub h0: T,
    t: T,
    y: V,
    h: T,
    pub rtol: T,
    pub atol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    // BS23 is not typically for stiff problems, so n_stiff related fields are omitted for simplicity.
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
    a: [[T; 4]; 3], // A has 3 rows for k2, k3, k4 based on k1, k2, k3
    b: [T; 3],      // For y_new (k1, k2, k3)
    c: [T; 3],      // For time points of k2, k3, k4
    er: [T; 4],     // For error estimation (k1, k2, k3, k4)
    k: [V; 4],      // k1, k2, k3, k4 (k4 is f(t+h, y_new))
    y_old: V,
    t_old: T,
    h_old: T,
    cont: [V; 4], // For cubic Hermite dense output
    cont_buffer: VecDeque<(T, T, T, [V; 4])>,
    phi: Option<H>,
    t0: T,
    tf: T,
    posneg: T,
    lags: [T; L], // Stores tau_i values
    yd: [V; L],   // Stores y(t-tau_i)
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData>
    DDENumericalMethod<L, T, V, H, D> for BS23<L, T, V, H, D>
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
                // Assuming phi is defined for t <= t0.
                // If t_delayed > t0 (for forward integration), it's an issue.
                // (t_delayed - self.t0) * self.posneg > T::zero() indicates t_delayed is "beyond" t0.
                if (t_delayed - self.t0) * self.posneg > T::default_epsilon() {
                    // Allow for t_delayed == t0
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
                3,
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
        self.h_old = self.h; // Can be zero if h0 was zero and h_init failed to produce non-zero.

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
        let k0_current_step_start = self.k[0]; // k1 for this step, from previous step's k4 (FSAL) or init

        let mut min_lag_abs = T::infinity();
        if L > 0 {
            let temp_y_for_lags = y_current_step_start + k0_current_step_start * self.h; // Estimate y at t+h
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

        let mut y_new_from_iter = y_current_step_start;
        let mut k_from_iter = self.k; // Stores k1, k2, k3, k4 of the converged iteration

        let mut last_y_for_errit_calc = V::zeros();
        let mut iteration_failed_to_converge = false;

        for iter_idx in 0..max_iter {
            if iter_idx > 0 {
                last_y_for_errit_calc = y_new_from_iter;
            }
            self.k[0] = k0_current_step_start; // k1

            // k2
            let mut ti = t_current_step_start + self.c[0] * self.h; // c[0] is for k2 (0.5)
            let mut yi = y_current_step_start + self.k[0] * (self.a[0][0] * self.h);
            if L > 0 {
                dde.lags(ti, &yi, &mut self.lags);
                self.lagvals(ti, &yi);
            }
            dde.diff(ti, &yi, &self.yd, &mut self.k[1]); // k2 stored in self.k[1]

            // k3
            ti = t_current_step_start + self.c[1] * self.h; // c[1] is for k3 (0.75)
            yi = y_current_step_start + self.k[1] * (self.a[1][1] * self.h); // BS23: y + h*A_32*k2
            if L > 0 {
                dde.lags(ti, &yi, &mut self.lags);
                self.lagvals(ti, &yi);
            }
            dde.diff(ti, &yi, &self.yd, &mut self.k[2]); // k3 stored in self.k[2]

            // y_new based on k1, k2, k3
            y_new_from_iter = y_current_step_start
                + (self.k[0] * self.b[0] + self.k[1] * self.b[1] + self.k[2] * self.b[2]) * self.h;

            // k4 (FSAL property, f(t+h, y_new))
            let t_new_val = t_current_step_start + self.h;
            // For BS23, A[2][...] are for k4. c[2] is for k4.
            // The k4 calculation in dde23 is f(tnew, ynew).
            if L > 0 {
                dde.lags(t_new_val, &y_new_from_iter, &mut self.lags);
                self.lagvals(t_new_val, &y_new_from_iter);
            }
            dde.diff(t_new_val, &y_new_from_iter, &self.yd, &mut self.k[3]); // k4 stored in self.k[3]

            evals.fcn += 3; // k2, k3, k4 evaluations
            k_from_iter.copy_from_slice(&self.k);

            if max_iter > 1 && iter_idx > 0 {
                let mut errit_val = T::zero();
                let n_dim = y_current_step_start.len();
                for i_dim in 0..n_dim {
                    let scale = self.atol
                        + self.rtol
                            * last_y_for_errit_calc
                                .get(i_dim)
                                .abs()
                                .max(y_new_from_iter.get(i_dim).abs());
                    if scale > T::zero() {
                        let diff_val =
                            y_new_from_iter.get(i_dim) - last_y_for_errit_calc.get(i_dim);
                        errit_val += (diff_val / scale).powi(2);
                    }
                }
                if n_dim > 0 {
                    errit_val = (errit_val / T::from_usize(n_dim).unwrap()).sqrt();
                }

                if errit_val <= self.rtol * T::from_f64(0.1).unwrap() {
                    break;
                }
            }
            if iter_idx == max_iter - 1 && max_iter > 1 {
                iteration_failed_to_converge = true;
            }
        }

        if iteration_failed_to_converge {
            self.h = (self.h.abs() * T::from_f64(0.5).unwrap()).max(self.h_min.abs()) * self.posneg;
            if L > 0
                && min_lag_abs > T::zero()
                && self.h.abs() < T::from_f64(2.0).unwrap() * min_lag_abs
            {
                self.h = min_lag_abs * self.posneg;
            }
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
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
            // Error = h * sum(E_i * k_i)
            let err_comp = k_from_iter[0].get(i) * self.er[0]
                + k_from_iter[1].get(i) * self.er[1]
                + k_from_iter[2].get(i) * self.er[2]
                + k_from_iter[3].get(i) * self.er[3];
            let erri = self.h * err_comp;
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

            // Dense output coefficients for cubic Hermite
            // y(s) = c0 + s(c1 + s(c2 + s*c3)) where s = (t_int - t_old)/h_old
            // c0 = y_old
            // c1 = h_old * k_old
            // c2 = 3*(y_new - y_old) - h_old*(2*k_old + k_new)
            // c3 = -2*(y_new - y_old) + h_old*(k_old + k_new)
            let k_old_for_cont = k_from_iter[0]; // k1 at t_current_step_start
            let k_new_for_cont = k_from_iter[3]; // k4 at t_new_val
            let y_diff_cont = y_new_from_iter - y_current_step_start;

            self.cont[0] = y_current_step_start;
            self.cont[1] = k_old_for_cont * self.h;
            self.cont[2] = y_diff_cont * T::from_f64(3.0).unwrap()
                - (k_old_for_cont * T::from_f64(2.0).unwrap() + k_new_for_cont) * self.h;
            self.cont[3] = y_diff_cont * T::from_f64(-2.0).unwrap()
                + (k_old_for_cont + k_new_for_cont) * self.h;

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

            self.k[0] = k_from_iter[3]; // FSAL: k4 becomes k1 for next step
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
    for BS23<L, T, V, H, D>
{
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // This interpolates within the last successfully completed step [t_old, t]
        if (t_interp - self.t_old) * self.posneg < T::zero()
            || (t_interp - self.t) * self.posneg > T::zero()
        {
            // If t_interp is exactly t_old or t, still proceed.
            if (t_interp - self.t_old).abs() > T::default_epsilon()
                && (t_interp - self.t).abs() > T::default_epsilon()
            {
                return Err(Error::OutOfBounds {
                    t_interp,
                    t_prev: self.t_old,
                    t_curr: self.t,
                });
            }
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

        // y_interp = cont[0] + s * (cont[1] + s * (cont[2] + s * cont[3]))
        let y_interp = self.cont[0] + (self.cont[1] + (self.cont[2] + self.cont[3] * s) * s) * s;
        Ok(y_interp)
    }
}

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> BS23<L, T, V, H, D> {
    pub fn new() -> Self {
        Self::default()
    }

    // Builder methods
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
        // y_stage not used if interpolating from buffer
        for i in 0..L {
            let t_delayed = t_stage - self.lags[i];
            if (t_delayed - self.t0) * self.posneg <= T::default_epsilon() {
                // t_delayed <= t0 (forward) or t_delayed >= t0 (backward)
                self.yd[i] = (self.phi.as_ref().unwrap())(t_delayed);
            } else {
                // Search cont_buffer (most recent first)
                let mut found_in_buffer = false;
                for (buf_t_start, buf_t_end, buf_h, buf_cont) in self.cont_buffer.iter().rev() {
                    // Check if t_delayed is within [buf_t_start, buf_t_end]
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
                        self.yd[i] =
                            buf_cont[0] + (buf_cont[1] + (buf_cont[2] + buf_cont[3] * s) * s) * s;
                        found_in_buffer = true;
                        break;
                    }
                }
                if !found_in_buffer {
                    // Extrapolate using the most recent interval if t_delayed is beyond it,
                    // or if buffer is empty (should ideally not happen if t_delayed > t0).
                    // This part might need more robust handling (e.g., error or specific extrapolation strategy)
                    if let Some((buf_t_start, _buf_t_end, buf_h, buf_cont)) =
                        self.cont_buffer.back()
                    {
                        let s = if *buf_h == T::zero() {
                            T::one()
                        } else {
                            (t_delayed - *buf_t_start) / *buf_h
                        }; // Extrapolation
                        self.yd[i] =
                            buf_cont[0] + (buf_cont[1] + (buf_cont[2] + buf_cont[3] * s) * s) * s;
                    } else {
                        // Fallback: if cont_buffer is empty and t_delayed > t0, this is an issue.
                        // This implies we are trying to access a future point not yet computed or history not covering it.
                        // For now, use phi, though this might be incorrect if t_delayed > t0.
                        // A proper error or specific handling for "future" lookups in early steps might be needed.
                        self.yd[i] = (self.phi.as_ref().unwrap())(t_delayed);
                        // Consider panic or error: panic!("Lag value lookup failed for t_delayed={}", t_delayed);
                    }
                }
            }
        }
    }
}

// Bogacki-Shampine Coefficients (FSAL variant)
// k1 = f(t,y)
// k2 = f(t+0.5h, y + 0.5h*k1)
// k3 = f(t+0.75h, y + 0.75h*k2)
// y_new = y + h*(2/9 k1 + 1/3 k2 + 4/9 k3)
// k4 = f(t+h, y_new)
// Error: y_new - y_hat where y_hat = y + h*(7/24 k1 + 1/4 k2 + 1/3 k3 + 1/8 k4)
// E = b_sol - b_err_coeffs = [2/9-7/24, 1/3-1/4, 4/9-1/3, 0-1/8]
//   = [-5/72, 1/12, 1/9, -1/8]

const BS23_C: [f64; 3] = [1.0 / 2.0, 3.0 / 4.0, 1.0]; // Time points for k2, k3, k4
const BS23_A: [[f64; 4]; 3] = [
    // A_ij * h * k_j
    [1.0 / 2.0, 0.0, 0.0, 0.0],             // k2 depends on k1
    [0.0, 3.0 / 4.0, 0.0, 0.0],             // k3 depends on k2
    [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0], // k4 (f(t+h, y_new)) uses y_new which depends on k1,k2,k3. This row is not directly used for y_i stages.
];
const BS23_B_SOL: [f64; 3] = [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0]; // For solution (uses k1, k2, k3)
const BS23_E: [f64; 4] = [-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0]; // For error estimate (uses k1, k2, k3, k4)

impl<const L: usize, T: Real, V: State<T>, H: Fn(T) -> V, D: CallBackData> Default
    for BS23<L, T, V, H, D>
{
    fn default() -> Self {
        let c_conv = BS23_C.map(|x| T::from_f64(x).unwrap());
        let a_conv = BS23_A.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b_sol_conv = BS23_B_SOL.map(|x| T::from_f64(x).unwrap());
        let er_conv = BS23_E.map(|x| T::from_f64(x).unwrap());

        let expo1_final = T::one() / T::from_f64(3.0).unwrap();

        let fac1_default = T::from_f64(0.2).unwrap(); // Can be tuned
        let fac2_default = T::from_f64(10.0).unwrap();

        BS23 {
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
            beta: T::zero(), // No PI by default for BS23
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
            b: b_sol_conv,
            c: c_conv,
            er: er_conv,
            k: [V::zeros(); 4],
            y_old: V::zeros(),
            t_old: T::zero(),
            h_old: T::zero(),
            cont: [V::zeros(); 4],
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
