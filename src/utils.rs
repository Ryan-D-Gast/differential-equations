//! Utility functions for the differential equation solvers

use crate::{
    error::Error,
    traits::{Real, State},
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
pub fn validate_step_size_parameters<T: Real, Y: State<T>>(
    h0: T,
    h_min: T,
    h_max: T,
    t0: T,
    tf: T,
) -> Result<T, Error<T, Y>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn test_constrain_step_size() {
        // Step within bounds
        assert_eq!(constrain_step_size(0.5, 0.1, 1.0), 0.5);
        assert_eq!(constrain_step_size(-0.5, 0.1, 1.0), -0.5);

        // Step too small (absolute value)
        assert_eq!(constrain_step_size(0.05, 0.1, 1.0), 0.1);
        assert_eq!(constrain_step_size(-0.05, 0.1, 1.0), -0.1);

        // Step too large (absolute value)
        assert_eq!(constrain_step_size(1.5, 0.1, 1.0), 1.0);
        assert_eq!(constrain_step_size(-1.5, 0.1, 1.0), -1.0);
    }

    #[test]
    fn test_validate_step_size_parameters_valid() {
        let result: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.1, 0.01, 1.0, 0.0, 10.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.1);

        let result_negative: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(-0.1, 0.01, 1.0, 10.0, 0.0);
        assert!(result_negative.is_ok());
        assert_eq!(result_negative.unwrap(), -0.1);
    }

    #[test]
    fn test_validate_step_size_parameters_tf_equals_t0() {
        let result: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.1, 0.01, 1.0, 5.0, 5.0);
        assert!(matches!(result, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result {
            assert!(msg.contains("cannot be equal to t0"));
        }
    }

    #[test]
    fn test_validate_step_size_parameters_wrong_sign() {
        // tf - t0 is positive, but h0 is negative
        let result: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(-0.1, 0.01, 1.0, 0.0, 10.0);
        assert!(matches!(result, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result {
            assert!(msg.contains("same sign as the integration direction"));
        }

        // tf - t0 is negative, but h0 is positive
        let result_negative: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.1, 0.01, 1.0, 10.0, 0.0);
        assert!(matches!(result_negative, Err(Error::BadInput { .. })));
    }

    #[test]
    fn test_validate_step_size_parameters_negative_bounds() {
        // h_min negative
        let result1: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.1, -0.01, 1.0, 0.0, 10.0);
        assert!(matches!(result1, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result1 {
            assert!(msg.contains("Minimum step size"));
            assert!(msg.contains("non-negative"));
        }

        // h_max negative
        let result2: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.1, 0.01, -1.0, 0.0, 10.0);
        assert!(matches!(result2, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result2 {
            assert!(msg.contains("Maximum step size"));
            assert!(msg.contains("non-negative"));
        }
    }

    #[test]
    fn test_validate_step_size_parameters_min_greater_than_max() {
        let result: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.5, 1.0, 0.1, 0.0, 10.0);
        assert!(matches!(result, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result {
            assert!(msg.contains("less than or equal to maximum step size"));
        }
    }

    #[test]
    fn test_validate_step_size_parameters_h0_out_of_bounds() {
        // |h0| < h_min
        let result1: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.001, 0.01, 1.0, 0.0, 10.0);
        assert!(matches!(result1, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result1 {
            assert!(msg.contains("greater than or equal to minimum step size"));
        }

        // |h0| > h_max
        let result2: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(2.0, 0.01, 1.0, 0.0, 10.0);
        assert!(matches!(result2, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result2 {
            assert!(msg.contains("less than or equal to maximum step size"));
        }

        // |h0| > |tf - t0|
        let result3: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.5, 0.01, 1.0, 0.0, 0.2);
        assert!(matches!(result3, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result3 {
            assert!(msg.contains("less than or equal to the absolute value of the integration interval"));
        }
    }

    #[test]
    fn test_validate_step_size_parameters_h0_zero() {
        let result: Result<f64, Error<f64, f64>> =
            validate_step_size_parameters(0.0, 0.0, 1.0, 0.0, 10.0);
        assert!(matches!(result, Err(Error::BadInput { .. })));
        if let Err(Error::BadInput { msg }) = result {
            assert!(msg.contains("cannot be zero"));
        }
    }
}
