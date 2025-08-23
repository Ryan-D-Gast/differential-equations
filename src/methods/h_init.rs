//! Initial step size picker

use crate::{
    dae::DAE,
    dde::DDE,
    linalg::Matrix,
    ode::ODE,
    stats::Evals,
    traits::{CallBackData, Real, State},
};

use super::{Algebraic, Delay, Ordinary};

/// Initial step size estimator using typestates for different equation types
pub struct InitialStepSize<Kind> {
    _phantom: std::marker::PhantomData<Kind>,
}

impl InitialStepSize<Ordinary> {
    /// Automatically compute an initial step size based on the ODE dynamics
    ///
    /// This function implements a robust algorithm to estimate an appropriate
    /// initial step size based on the derivatives and their rates of change.
    /// It uses the order of the method and error tolerances to compute a step
    /// size that should yield accurate results.
    ///
    /// Note uses 2 function evaluations to compute the initial step size.
    ///
    /// # Arguments
    ///
    /// * `ode` - The ODE function to estimate initial step size for
    /// * `t0` - The initial time
    /// * `tf` - The final time (used to determine direction)
    /// * `y0` - The initial state
    /// * `order` - The order of the numerical method
    /// * `rtol` - Relative tolerance
    /// * `atol` - Absolute tolerance    
    /// * `h_min` - Minimum allowed step size
    /// * `h_max` - Maximum allowed step size
    /// * `evals` - Function evaluation counter
    ///
    /// # Returns
    ///
    /// The estimated initial step size
    pub fn compute<T, F, Y, D>(
        ode: &F,
        t0: T,
        tf: T,
        y0: &Y,
        order: usize,
        rtol: T,
        atol: T,
        h_min: T,
        h_max: T,
        evals: &mut Evals,
    ) -> T
    where
        T: Real,
        Y: State<T>,
        F: ODE<T, Y, D>,
        D: CallBackData,
    {
        // Direction of integration
        let posneg = (tf - t0).signum();

        // Storage for derivatives
        let mut f0 = Y::zeros();
        let mut f1 = Y::zeros(); // Compute initial derivative f(t0, y0)
        ode.diff(t0, y0, &mut f0);
        evals.function += 1;

        // Compute weighted norm of the initial derivative and solution
        let mut dnf = T::zero();
        let mut dny = T::zero();

        // Loop through all elements to compute weighted norms
        for n in 0..y0.len() {
            let sk = atol + rtol * y0.get(n).abs();
            dnf += (f0.get(n) / sk).powi(2);
            dny += (y0.get(n) / sk).powi(2);
        }

        // Initial step size guess
        let mut h: T;
        if dnf <= T::from_f64(1.0e-10).unwrap() || dny <= T::from_f64(1.0e-10).unwrap() {
            h = T::from_f64(1.0e-6).unwrap();
        } else {
            h = (dny / dnf).sqrt() * T::from_f64(0.01).unwrap();
        }

        // Constrain by maximum step size
        h = h.min(h_max);
        h *= posneg;

        // Perform an explicit Euler step
        let y1 = *y0 + f0 * h; // Evaluate derivative at new point
        ode.diff(t0 + h, &y1, &mut f1);
        evals.function += 1;

        // Estimate the second derivative
        let mut der2 = T::zero();

        for n in 0..y0.len() {
            let sk = atol + rtol * y0.get(n).abs();
            der2 += ((f1.get(n) - f0.get(n)) / sk).powi(2);
        }
        der2 = der2.sqrt() / h.abs();

        // Calculate step size based on order and error constraints
        // h^order * max(|f0|, |der2|) = 0.01
        let der12 = dnf.sqrt().max(der2);

        let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
            h.abs()
                * T::from_f64(1.0e-3)
                    .unwrap()
                    .max(T::from_f64(1.0e-6).unwrap())
        } else {
            // Convert order to T
            let order_t = T::from_usize(order).unwrap();
            (T::from_f64(0.01).unwrap() / der12).powf(T::one() / order_t)
        };

        // Final bounds checking (also cap by the interval length)
        let interval = (tf - t0).abs();
        h = (h.abs() * T::from_f64(100.0).unwrap())
            .min(h1)
            .min(h_max)
            .max(h_min)
            .min(interval);
        h * posneg
    }
}

impl InitialStepSize<Delay> {
    /// Automatically compute an initial step size for DDEs
    ///
    /// This function implements a robust algorithm to estimate an appropriate
    /// initial step size for delay differential equations, taking into account
    /// the delay structure and ensuring that the initial step doesn't violate
    /// the delay conditions.
    ///
    /// # Arguments
    ///
    /// * `dde` - The DDE function to estimate initial step size for
    /// * `t0` - The initial time
    /// * `tf` - The final time (used to determine direction)
    /// * `y0` - The initial state
    /// * `order` - The order of the numerical method
    /// * `rtol` - Relative tolerance
    /// * `atol` - Absolute tolerance
    /// * `h_min` - Minimum allowed step size
    /// * `h_max` - Maximum allowed step size
    /// * `phi` - History function for delayed values
    /// * `f0` - Initial derivative value
    /// * `evals` - Function evaluation counter
    ///
    /// # Returns
    ///
    /// The estimated initial step size
    pub fn compute<const L: usize, T, Y, D, F>(
        dde: &F,
        t0: T,
        tf: T,
        y0: &Y,
        order: usize,
        rtol: T,
        atol: T,
        h_min: T,
        h_max: T,
        phi: &impl Fn(T) -> Y,
        f0: &Y,
        evals: &mut Evals,
    ) -> T
    where
        T: Real,
        Y: State<T>,
        D: CallBackData,
        F: DDE<L, T, Y, D>,
    {
        let posneg_init = (tf - t0).signum();
        let n_dim = y0.len();

        let mut dnf = T::zero();
        let mut dny = T::zero();
        for n in 0..n_dim {
            let sk = atol + rtol * y0.get(n).abs();
            if sk <= T::zero() {
                return h_min.abs().max(T::from_f64(1e-6).unwrap()) * posneg_init;
            }
            dnf += (f0.get(n) / sk).powi(2);
            dny += (y0.get(n) / sk).powi(2);
        }
        if n_dim > 0 {
            dnf = (dnf / T::from_usize(n_dim).unwrap()).sqrt();
            dny = (dny / T::from_usize(n_dim).unwrap()).sqrt();
        } else {
            // Scalar case
            dnf = dnf.sqrt();
            dny = dny.sqrt();
        }

        let mut h = if dnf <= T::from_f64(1.0e-10).unwrap() || dny <= T::from_f64(1.0e-10).unwrap()
        {
            T::from_f64(1.0e-6).unwrap()
        } else {
            (dny / dnf) * T::from_f64(0.01).unwrap()
        };
        h = h.min(h_max.abs());
        h *= posneg_init;

        let mut y1 = *y0 + *f0 * h;
        let mut t1 = t0 + h;
        let mut f1 = Y::zeros();

        let mut current_lags_init = [T::zero(); L];
        let mut yd_init = [Y::zeros(); L];

        // Ensure initial step's delayed points are valid
        if L > 0 {
            loop {
                // Adjust h if t1 - lag is "beyond" t0 for phi
                dde.lags(t1, &y1, &mut current_lags_init);
                let mut reduce_h_for_lag = false;
                let mut h_candidate_from_lag = h.abs();

                for i in 0..L {
                    if current_lags_init[i] <= T::zero() {
                        /* error or skip */
                        continue;
                    }
                    let t_delayed = t1 - current_lags_init[i];
                    if (t_delayed - t0) * posneg_init < -T::default_epsilon() { // t_delayed is "before" t0
                        // This is fine, phi will be used.
                    } else {
                        // t_delayed is "at or after" t0. This means current h is too large.
                        // We need t1 - lag <= t0  => h + t0 - lag <= t0 => h <= lag
                        h_candidate_from_lag = h_candidate_from_lag
                            .min(current_lags_init[i].abs() * T::from_f64(0.99).unwrap()); // Reduce h to be less than this lag
                        reduce_h_for_lag = true;
                    }
                }

                if reduce_h_for_lag && h_candidate_from_lag < h.abs() {
                    h = h_candidate_from_lag * posneg_init;
                    if h.abs() < h_min.abs() {
                        h = h_min * posneg_init;
                    } // Respect h_min
                    if h.abs() < T::default_epsilon() {
                        // Avoid zero step
                        return h_min.abs().max(T::from_f64(1e-6).unwrap()) * posneg_init;
                    }
                    y1 = *y0 + *f0 * h;
                    t1 = t0 + h;
                    // Loop again with new h
                } else {
                    break; // h is fine regarding lags for phi
                }
            }
            // Populate yd_init for the diff call
            dde.lags(t1, &y1, &mut current_lags_init); // Recalculate lags with final t1, y1
            for i in 0..L {
                let t_delayed = t1 - current_lags_init[i];
                yd_init[i] = phi(t_delayed);
            }
        }

        dde.diff(t1, &y1, &yd_init, &mut f1);
        evals.function += 1;

        let mut der2 = T::zero();
        for n in 0..n_dim {
            let sk = atol + rtol * y0.get(n).abs();
            if sk <= T::zero() {
                der2 = T::infinity();
                break;
            }
            der2 += ((f1.get(n) - f0.get(n)) / sk).powi(2);
        }
        if n_dim > 0 {
            der2 = (der2 / T::from_usize(n_dim).unwrap()).sqrt() / h.abs();
        } else {
            // Scalar
            der2 = der2.sqrt() / h.abs();
        }

        let der12 = dnf.max(der2);
        let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
            h.abs().max(T::from_f64(1.0e-6).unwrap()) * T::from_f64(0.1).unwrap()
        } else {
            let order_t = T::from_usize(order + 1).unwrap(); // order is method order (3 for BS23)
            (T::from_f64(0.01).unwrap() / der12).powf(T::one() / order_t)
        };

        h = h.abs().min(h1);
        h = h.min(h_max.abs());
        if h_min.abs() > T::zero() {
            h = h.max(h_min.abs());
        }
        h = h.min((tf - t0).abs());
        h * posneg_init
    }
}

impl InitialStepSize<Algebraic> {
    /// Compute an initial step size for DAEs using the mass matrix structure.
    ///
    /// Heuristic mirrors the ODE estimator but only measures derivative norms on
    /// differential components (rows of M with non-negligible entries). We avoid
    /// inverting the mass matrix; algebraic rows are ignored in derivative norms.
    pub fn compute<T, F, Y, D>(
        dae: &F,
        t0: T,
        tf: T,
        y0: &Y,
        order: usize,
        rtol: T,
        atol: T,
        h_min: T,
        h_max: T,
        evals: &mut Evals,
    ) -> T
    where
        T: Real,
        Y: State<T>,
        F: DAE<T, Y, D>,
        D: CallBackData,
    {
        let posneg = (tf - t0).signum();
        let dim = y0.len();

        // Mass matrix to determine which equations are differential vs algebraic
        let mut m = Matrix::zeros(dim, dim);
        dae.mass(&mut m);
        let eps = T::from_f64(1e-14).unwrap();
        let mut is_diff = vec![false; dim];
        for i in 0..dim {
            let mut row_sum = T::zero();
            for j in 0..dim {
                row_sum = row_sum + m[(i, j)].abs();
            }
            is_diff[i] = row_sum > eps;
        }

        // f(t0, y0)
        let mut f0 = Y::zeros();
        dae.diff(t0, y0, &mut f0);
        evals.function += 1;

        // Weighted norms (derivative norm only on differential rows)
        let mut dnf = T::zero();
        let mut dny = T::zero();
        for n in 0..dim {
            let sk = atol + rtol * y0.get(n).abs();
            dny = dny + (y0.get(n) / sk).powi(2);
            if is_diff[n] {
                dnf = dnf + (f0.get(n) / sk).powi(2);
            }
        }

        // If there are no differential equations, fall back to a tiny step
        if !is_diff.iter().any(|&b| b) {
            return h_min.abs().max(T::from_f64(1e-6).unwrap()) * posneg;
        }

        // Initial step guess
        let mut h: T =
            if dnf <= T::from_f64(1.0e-10).unwrap() || dny <= T::from_f64(1.0e-10).unwrap() {
                T::from_f64(1.0e-6).unwrap()
            } else {
                (dny / dnf).sqrt() * T::from_f64(0.01).unwrap()
            };

        // Bound and sign
        h = h.min(h_max.abs());
        h = h.max(h_min.abs());
        h = h * posneg;

        // One explicit Euler predictor (uses f0 directly; avoids M^{-1})
        let y1 = *y0 + f0 * h;
        let mut f1 = Y::zeros();
        dae.diff(t0 + h, &y1, &mut f1);
        evals.function += 1;

        // Second derivative estimate on differential components only
        let mut der2 = T::zero();
        for n in 0..dim {
            if !is_diff[n] {
                continue;
            }
            let sk = atol + rtol * y0.get(n).abs();
            der2 = der2 + ((f1.get(n) - f0.get(n)) / sk).powi(2);
        }
        der2 = der2.sqrt() / h.abs().max(T::default_epsilon());

        // Use method order to scale an h that enforces ~1% local error
        let der12 = dnf.sqrt().max(der2);
        let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
            h.abs()
                * T::from_f64(1.0e-3)
                    .unwrap()
                    .max(T::from_f64(1.0e-6).unwrap())
        } else {
            let order_t = T::from_usize(order).unwrap();
            (T::from_f64(0.01).unwrap() / der12).powf(T::one() / order_t)
        };

        // Final bounds (cap by interval length as well)
        let mut h_final = (h.abs() * T::from_f64(100.0).unwrap())
            .min(h1)
            .min(h_max.abs());
        if h_min.abs() > T::zero() {
            h_final = h_final.max(h_min.abs());
        }
        h_final = h_final.min((tf - t0).abs());
        h_final * posneg
    }
}
