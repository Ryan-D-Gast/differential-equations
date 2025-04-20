//! Initial step size picker

use crate::{
    ode::ODE,
    traits::{CallBackData, Real, State},
};

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
///
/// # Returns
///
/// The estimated initial step size
///
pub fn h_init<T, F, V, D>(
    ode: &F,
    t0: T,
    tf: T,
    y0: &V,
    order: usize,
    rtol: T,
    atol: T,
    h_min: T,
    h_max: T,
) -> T
where
    T: Real,
    V: State<T>,
    F: ODE<T, V, D>,
    D: CallBackData,
{
    // Direction of integration
    let posneg = (tf - t0).signum();

    // Storage for derivatives
    let mut f0 = V::zeros();
    let mut f1 = V::zeros();

    // Compute initial derivative f(t0, y0)
    ode.diff(t0, y0, &mut f0);

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
    let y1 = *y0 + f0 * h;

    // Evaluate derivative at new point
    ode.diff(t0 + h, &y1, &mut f1);

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

    // Final bounds checking
    h = (h.abs() * T::from_f64(100.0).unwrap())
        .min(h1)
        .min(h_max)
        .max(h_min);

    // Return with proper sign
    h * posneg
}
