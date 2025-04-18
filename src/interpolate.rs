//! Interpolation Methods for the IVP struct when solving the system.

use crate::traits::Real;
use nalgebra::SMatrix;
use std::fmt::{Debug, Display};

/// Interpolation trait implemented by Solvers to allow Solout to access interpolated values between t_prev and t_curr
pub trait Interpolation<T, const R: usize, const C: usize> 
where 
    T: Real,
{
    /// Interpolate between previous and current step
    /// 
    /// Note that the range for interpolation is between t_prev and t_curr.
    /// If t_interp is outside this range, an error will be returned in the
    /// form of an InterpolationError::OutOfBounds.
    ///
    /// # Arguments
    /// * `t_interp`  - Time to interpolate at.
    ///
    /// # Returns
    /// * Interpolated State Vector.
    ///
    fn interpolate(
        &mut self,
        t_interp: T,
    ) -> Result<SMatrix<T, R, C>, InterpolationError<T, R, C>>;
}

/// Interpolation Error for ODE NumericalMethods
///
/// # Variants
/// * `OutOfBounds` - Given t is not within the previous and current step.
///
#[derive(PartialEq, Clone)]
pub enum InterpolationError<T, const R: usize, const C: usize>
where
    T: Real,
{
    /// Given t is not within the previous and current step
    OutOfBounds {
        t_interp: T,
        t_prev: T,
        t_curr: T,
    }
}

impl<T, const R: usize, const C: usize> Display for InterpolationError<T, R, C>
where
    T: Real,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationError::OutOfBounds { t_interp, t_prev, t_curr } => {
                write!(
                    f,
                    "Interpolation Error: t_interp {} is not within the previous and current step: t_prev {}, t_curr {}",
                    t_interp, t_prev, t_curr
                )
            }
        }
    }
}

impl<T, const R: usize, const C: usize> Debug for InterpolationError<T, R, C>
where
    T: Real,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationError::OutOfBounds { t_interp, t_prev, t_curr } => {
                write!(
                    f,
                    "Interpolation Error: t_interp {} is not within the previous and current step: t_prev {}, t_curr {}",
                    t_interp, t_prev, t_curr
                )
            }
        }
    }
}

impl<T, const R: usize, const C: usize> std::error::Error for InterpolationError<T, R, C>
where
    T: Real,
{}

/// Cubic Hermite Interpolation
///
/// # Arguments
/// * `t0` - Initial Time.
/// * `t1` - Final Time.
/// * `y0` - Initial State Vector.
/// * `y1` - Final State Vector.
/// * `t`  - Time to interpolate at.
///
/// # Returns
/// * Interpolated State Vector.
///
pub fn cubic_hermite_interpolate<T: Real, const R: usize, const C: usize>(
    t0: T,
    t1: T,
    y0: &SMatrix<T, R, C>,
    y1: &SMatrix<T, R, C>,
    k0: &SMatrix<T, R, C>,
    k1: &SMatrix<T, R, C>,
    t: T,
) -> SMatrix<T, R, C> {
    let two = T::from_f64(2.0).unwrap();
    let three = T::from_f64(3.0).unwrap();
    let h = t1 - t0;
    let s = (t - t0) / h;
    let h00 = two * s.powi(3) - three * s.powi(2) + T::one();
    let h10 = s.powi(3) - two * s.powi(2) + s;
    let h01 = -two * s.powi(3) + three * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);
    y0 * h00 + k0 * h10 * h + y1 * h01 + k1 * h11 * h
}