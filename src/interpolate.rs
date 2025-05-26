//! Interpolation Methods for the ODEProblem struct when solving the system.

use crate::{
    error::Error,
    traits::{Real, State},
};

/// Interpolation trait implemented by Solvers to allow Solout to access interpolated values between t_prev and t_curr
pub trait Interpolation<T, V>
where
    T: Real,
    V: State<T>,
{
    /// Interpolate between previous and current step
    ///
    /// Note that the range for interpolation is between t_prev and t_curr.
    /// If t_interp is outside this range, an error will be returned in the
    /// form of an Error::OutOfBounds.
    ///
    /// # Arguments
    /// * `t_interp`  - Time to interpolate at.
    ///
    /// # Returns
    /// * Interpolated State Vector.
    ///
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>>;
}

/// Cubic Hermite Interpolation
///
/// # Arguments
/// * `t0` - Initial Time.
/// * `t1` - Final Time.
/// * `y0` - Initial State Vector.
/// * `y1` - Final State Vector.
/// * `k0` - Initial Deriv of State Vector.
/// * `k1` - Final Deriv of State Vector.
/// * `t`  - Time to interpolate at.
///
/// # Returns
/// * Interpolated State Vector.
///
pub fn cubic_hermite_interpolate<T: Real, V: State<T>>(
    t0: T,
    t1: T,
    y0: &V,
    y1: &V,
    k0: &V,
    k1: &V,
    t: T,
) -> V {
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
