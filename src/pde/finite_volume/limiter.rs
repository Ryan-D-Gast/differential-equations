//! Limiters for finite volume reconstruction.

use crate::traits::Real;

/// Slope limiter for piecewise linear reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Limiter {
    /// No limiting (unstable near shocks).
    None,
    /// Minmod limiter (most dissipative).
    #[default]
    Minmod,
    /// Superbee limiter (least dissipative).
    Superbee,
    /// Van Leer limiter (smooth).
    VanLeer,
}

impl Limiter {
    /// Compute the limited slope ratio.
    pub fn compute<T: Real>(&self, r: T) -> T {
        match self {
            Self::None => T::one(),
            Self::Minmod => {
                let zero = T::zero();
                let one = T::one();
                if r > zero { r.min(one) } else { zero }
            }
            Self::Superbee => {
                let zero = T::zero();
                let one = T::one();
                let two = T::from_subset(&2.0);
                if r > zero {
                    let min1 = r.min(two);
                    let min2 = (two * r).min(one);
                    min1.max(min2)
                } else {
                    zero
                }
            }
            Self::VanLeer => {
                let zero = T::zero();
                if r > zero {
                    let num = r + r.abs();
                    let den = T::one() + r.abs();
                    num / den
                } else {
                    zero
                }
            }
        }
    }
}
