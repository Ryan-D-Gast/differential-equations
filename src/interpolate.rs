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
