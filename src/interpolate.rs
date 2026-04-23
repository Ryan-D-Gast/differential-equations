//! Interpolation utilities used by solvers and output handlers.

use crate::{
    error::Error,
    traits::{Real, State},
};

/// Interpolation capability provided by solvers for dense output.
///
/// Implementations expose an interpolant valid on the current step
/// interval `[t_prev, t_curr]` so that downstream components (e.g.,
/// `Solout`) can query states between accepted steps.
pub trait Interpolation<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// Evaluate the step-local interpolant at the given time.
    ///
    /// The valid domain is the current step interval `[t_prev, t_curr]`.
    /// If `t_interp` lies outside this range, an `Error::OutOfBounds` is
    /// returned.
    ///
    /// - Input: `t_interp` time within the current step
    /// - Output: interpolated state `Y` or an error
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>>;
}

/// Cubic Hermite interpolation over `[t0, t1]`.
///
/// Uses endpoint values and derivatives to construct a smooth
/// cubic interpolant.
///
/// - Inputs:
///   - `t0`, `t1`: interval bounds
///   - `y0`, `y1`: state values at `t0`, `t1`
///   - `k0`, `k1`: state derivatives at `t0`, `t1`
///   - `t`: evaluation time
/// - Output: interpolated state at `t`
pub fn cubic_hermite_interpolate<T: Real, Y: State<T>>(
    t0: T,
    t1: T,
    y0: &Y,
    y1: &Y,
    k0: &Y,
    k1: &Y,
    t: T,
) -> Y {
    let two = T::from_f64(2.0).unwrap();
    let three = T::from_f64(3.0).unwrap();
    let h = t1 - t0;
    let s = (t - t0) / h;
    let h00 = two * s.powi(3) - three * s.powi(2) + T::one();
    let h10 = s.powi(3) - two * s.powi(2) + s;
    let h01 = -two * s.powi(3) + three * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);
    *y0 * h00 + *k0 * h10 * h + *y1 * h01 + *k1 * h11 * h
}

/// Linear interpolation over `[t0, t1]`.
///
/// Computes the straight-line interpolant between `y0` and `y1`.
///
/// - Inputs:
///   - `t0`, `t1`: interval bounds
///   - `y0`, `y1`: state values at `t0`, `t1`
///   - `t`: evaluation time
/// - Output: interpolated state at `t`
pub fn linear_interpolate<T: Real, Y: State<T>>(t0: T, t1: T, y0: &Y, y1: &Y, t: T) -> Y {
    let s = (t - t0) / (t1 - t0);
    *y0 * (T::one() - s) + *y1 * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;

    #[test]
    fn test_linear_interpolate_f64() {
        let t0 = 0.0_f64;
        let t1 = 1.0_f64;
        let y0 = 0.0_f64;
        let y1 = 2.0_f64;

        assert!((linear_interpolate(t0, t1, &y0, &y1, t0) - y0).abs() < 1e-10);
        assert!((linear_interpolate(t0, t1, &y0, &y1, t1) - y1).abs() < 1e-10);
        assert!((linear_interpolate(t0, t1, &y0, &y1, 0.5_f64) - 1.0_f64).abs() < 1e-10);
        assert!((linear_interpolate(t0, t1, &y0, &y1, 2.0_f64) - 4.0_f64).abs() < 1e-10);
        assert!((linear_interpolate(t0, t1, &y0, &y1, -1.0_f64) - (-2.0_f64)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolate_smatrix() {
        let t0 = 0.0;
        let t1 = 2.0;
        let y0 = SMatrix::<f64, 2, 1>::new(1.0, 2.0);
        let y1 = SMatrix::<f64, 2, 1>::new(3.0, 6.0);

        let res_t0 = linear_interpolate(t0, t1, &y0, &y1, t0);
        assert!((res_t0 - y0).norm() < 1e-10);

        let res_t1 = linear_interpolate(t0, t1, &y0, &y1, t1);
        assert!((res_t1 - y1).norm() < 1e-10);

        let expected_mid = SMatrix::<f64, 2, 1>::new(2.0, 4.0);
        let res_mid = linear_interpolate(t0, t1, &y0, &y1, 1.0);
        assert!((res_mid - expected_mid).norm() < 1e-10);
    }

    #[test]
    fn test_cubic_hermite_interpolate_bounds() {
        let t0 = 0.0;
        let t1 = 1.0;
        let y0 = 2.5;
        let y1 = -1.5;
        let k0 = 1.0;
        let k1 = -0.5;

        // At t0, should evaluate exactly to y0
        let res0 = cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t0);
        assert_eq!(res0, y0);

        // At t1, should evaluate exactly to y1
        let res1 = cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t1);
        assert_eq!(res1, y1);
    }

    #[test]
    fn test_cubic_hermite_interpolate_analytical() {
        // Test exact reproduction of a cubic polynomial f(t) = t^3.
        // f(t) = t^3
        // f'(t) = 3t^2
        let t0 = 1.0;
        let t1 = 2.0;
        let y0 = 1.0; // 1^3
        let y1 = 8.0; // 2^3
        let k0 = 3.0; // 3 * 1^2
        let k1 = 12.0; // 3 * 2^2

        let t_mid = 1.5;
        let expected_mid = 3.375; // 1.5^3

        let res_mid: f64 = cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t_mid);
        assert!(
            (res_mid - expected_mid).abs() < 1e-12,
            "Expected {}, got {}",
            expected_mid,
            res_mid
        );

        let t_quarter = 1.25;
        let expected_quarter = 1.953125; // 1.25^3

        let res_quarter: f64 = cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t_quarter);
        assert!(
            (res_quarter - expected_quarter).abs() < 1e-12,
            "Expected {}, got {}",
            expected_quarter,
            res_quarter
        );
    }

    #[test]
    fn test_cubic_hermite_interpolate_f64() {
        let t0 = 0.0_f64;
        let t1 = 1.0_f64;
        let y0 = 0.0_f64;
        let y1 = 1.0_f64;
        let k0 = 0.0_f64;
        let k1 = 0.0_f64;

        assert!((cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t0) - y0).abs() < 1e-10);
        assert!((cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t1) - y1).abs() < 1e-10);
        assert!(
            (cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, 0.5_f64) - 0.5_f64).abs()
                < 1e-10
        );

        let k0 = 1.0_f64;
        let k1 = 1.0_f64;
        assert!(
            (cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, 0.5_f64) - 0.5_f64).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_cubic_hermite_interpolate_smatrix() {
        let t0 = 0.0;
        let t1 = 1.0;
        let y0 = SMatrix::<f64, 1, 1>::new(0.0);
        let y1 = SMatrix::<f64, 1, 1>::new(1.0);
        let k0 = SMatrix::<f64, 1, 1>::new(1.0);
        let k1 = SMatrix::<f64, 1, 1>::new(1.0);

        let res = cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, 0.5);
        assert!((res[(0, 0)] - 0.5).abs() < 1e-10);
    }
}
