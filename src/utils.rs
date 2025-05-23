use crate::{
    Error,
    traits::{CallBackData, Real, State},
};

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
/// If any of the checks fail, returns `Err(Error::BadInput)` with a descriptive message.
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
/// * `Result<Real, Error>` - Ok if all checks pass, Err if any check fails.
///
pub fn validate_step_size_parameters<T: Real, V: State<T>, D: CallBackData>(
    h0: T,
    h_min: T,
    h_max: T,
    t0: T,
    tf: T,
) -> Result<T, Error<T, V>> {
    // Check if tf == t0
    if tf == t0 {
        return Err(Error::BadInput {
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
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Initial step size ({:?}) must have the same sign as the integration direction (sign of tf - t0 = {:?})",
                h0,
                tf - t0
            ),
        });
    }

    // Check h_min and h_max bounds
    if h_min < T::zero() {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Minimum step size ({:?}) must be non-negative",
                h_min
            ),
        });
    }
    if h_max < T::zero() {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Maximum step size ({:?}) must be non-negative",
                h_max
            ),
        });
    }
    if h_min > h_max {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Minimum step size ({:?}) must be less than or equal to maximum step size ({:?})",
                h_min, h_max
            ),
        });
    }

    // Check h0 bounds
    if h0.abs() < h_min {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be greater than or equal to minimum step size ({:?})",
                h0.abs(),
                h_min
            ),
        });
    }
    if h0.abs() > h_max {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be less than or equal to maximum step size ({:?})",
                h0.abs(),
                h_max
            ),
        });
    }

    // Check h0 is not larger then integration interval
    if h0.abs() > (tf - t0).abs() {
        return Err(Error::BadInput {
            msg: format!(
                "Invalid input: Absolute value of initial step size ({:?}) must be less than or equal to the absolute value of the integration interval (tf - t0 = {:?})",
                h0.abs(),
                (tf - t0).abs()
            ),
        });
    }

    // Check h0 is not zero
    if h0 == T::zero() {
        return Err(Error::BadInput {
            msg: format!("Invalid input: Initial step size ({:?}) cannot be zero", h0),
        });
    }

    // Return Ok if all bounds are valid return the step size
    Ok(h0)
}
