use crate::ode::{CallBackData, SolverError};
use crate::traits::Real;

/// Validate the step size parameters.
///
/// Checks the following:
/// * `tf` cannot be equal to `t0`.
/// * `h0` has the same sign as `tf - t0`.
/// * `h_min` and `h_max` are non-negative.
/// * `h_min` is less than or equal to `h_max`.
/// * `|h0|` is greater than or equal to `h_min`.
/// * `|h0|` is less than or equal to `h_max`.
/// * `|h0|` is less than or equal to `|tf - t0|`.
/// * `h0` is not zero.
///
/// If any of the checks fail, returns `Err(SolverError::BadInput)` with a descriptive message.
/// Else returns `Ok(h0)` indicating the step size is valid.
///
/// # Arguments
/// * `h0` - Initial step size.
/// * `h_min` - Minimum step size.
/// * `h_max` - Maximum step size.
/// * `t0` - Initial time.
/// * `tf` - Final time.
///
/// # Returns
/// * `Result<Real, SolverError>` - Ok if all checks pass, Err if any check fails.
///
pub fn validate_step_size_parameters<T: Real, const R: usize, const C: usize, D: CallBackData>(
    h0: T,
    h_min: T,
    h_max: T,
    t0: T,
    tf: T,
) -> Result<T, SolverError<T, R, C>> {
    // Check if tf == t0
    if tf == t0 {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: tf ({:?}) cannot be equal to t0 ({:?})",
                tf, t0
            ),
        });
    }

    // Determine direction of the step size
    let sign = (tf - t0).signum();

    // Check h0 has same sign as tf - t0
    if h0.signum() != sign {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Initial step size ({:?}) must have the same sign as the integration direction (sign of tf - t0 = {:?})",
                h0,
                tf - t0
            ),
        });
    }

    // Check h_min and h_max bounds
    if h_min < T::zero() {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Minimum step size ({:?}) must be non-negative",
                h_min
            ),
        });
    }
    if h_max < T::zero() {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Maximum step size ({:?}) must be non-negative",
                h_max
            ),
        });
    }
    if h_min > h_max {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Minimum step size ({:?}) must be less than or equal to maximum step size ({:?})",
                h_min, h_max
            ),
        });
    }

    // Check h0 bounds
    if h0.abs() < h_min {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be greater than or equal to minimum step size ({:?})",
                h0.abs(),
                h_min
            ),
        });
    }
    if h0.abs() > h_max {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be less than or equal to maximum step size ({:?})",
                h0.abs(),
                h_max
            ),
        });
    }

    // Check h0 is not larger then integration interval
    if h0.abs() > (tf - t0).abs() {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be less than or equal to the absolute value of the integration interval (tf - t0 = {:?})",
                h0.abs(),
                (tf - t0).abs()
            ),
        });
    }

    // Check h0 is not zero
    if h0 == T::zero() {
        return Err(SolverError::BadInput {
            msg: format!(
                "Invalid input: Initial step size ({:?}) cannot be zero",
                h0
            ),
        });
    }

    // Return Ok if all bounds are valid return the step size
    Ok(h0)
}

/// Constrain the step size to be within the bounds of `h_min` and `h_max`.
///
/// If `h` is less than `h_min`, returns `h_min` with the same sign as `h`.
/// If `h` is greater than `h_max`, returns `h_max` with the same sign as `h`.
/// Otherwise, returns `h` unchanged.
///
/// # Arguments
/// * `h` - Step size to constrain.
/// * `h_min` - Minimum step size.
/// * `h_max` - Maximum step size.
///
/// # Returns
/// * The step size constrained to be within the bounds of `h_min` and `h_max`.
///
pub fn constrain_step_size<T: Real>(h: T, h_min: T, h_max: T) -> T {
    // Determine the direction of the step size
    let sign = h.signum();
    // Bound the step size
    if h.abs() < h_min {
        sign * h_min
    } else if h.abs() > h_max {
        sign * h_max
    } else {
        h
    }
}

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
pub fn h_init<T, F, const R: usize, const C: usize, D>(
    ode: &F,
    t0: T,
    tf: T,
    y0: &crate::SMatrix<T, R, C>,
    order: usize,
    rtol: T,
    atol: T,
    h_min: T,
    h_max: T,
) -> T
where
    T: crate::traits::Real,
    F: crate::ode::ODE<T, R, C, D>,
    D: crate::ode::CallBackData,
{
    // Direction of integration
    let posneg = (tf - t0).signum();

    // Storage for derivatives
    let mut f0 = crate::SMatrix::<T, R, C>::zeros();
    let mut f1 = crate::SMatrix::<T, R, C>::zeros();

    // Compute initial derivative f(t0, y0)
    ode.diff(t0, y0, &mut f0);

    // Compute weighted norm of the initial derivative and solution
    let mut dnf = T::zero();
    let mut dny = T::zero();

    // Loop through all elements to compute weighted norms
    for r in 0..R {
        for c in 0..C {
            let sk = atol + rtol * y0[(r, c)].abs();
            dnf += (f0[(r, c)] / sk).powi(2);
            dny += (y0[(r, c)] / sk).powi(2);
        }
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
    let y1 = y0 + f0 * h;

    // Evaluate derivative at new point
    ode.diff(t0 + h, &y1, &mut f1);

    // Estimate the second derivative
    let mut der2 = T::zero();

    for r in 0..R {
        for c in 0..C {
            let sk = atol + rtol * y0[(r, c)].abs();
            der2 += ((f1[(r, c)] - f0[(r, c)]) / sk).powi(2);
        }
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
