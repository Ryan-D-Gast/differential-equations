//! Numerical Methods for Differential Equations

mod h_init;

// --- Explicit Runge-Kutta Methods ---
mod erk;
pub use erk::ExplicitRungeKutta;

// --- Implicit Runge-Kutta Methods ---
mod irk;
pub use irk::ImplicitRungeKutta;

// --- Diagonally Implicit Runge-Kutta Methods ---
mod dirk;
pub use dirk::DiagonallyImplicitRungeKutta;

// --- Adams Predictor-Corrector Methods ---
mod apc;
pub use apc::AdamsPredictorCorrector;

// --- Typestate Categories for Differential Equations Types ---
pub struct Ordinary;
pub struct Delay;
pub struct Stochastic;
pub struct Algebraic;

// --- Typestate Categories for Numerical Methods Families ---

/// Fixed-step methods
pub struct Fixed;

/// Adaptive-step methods
pub struct Adaptive;

/// Explicit Adaptive-step methods by Dormand-Prince
pub struct DormandPrince;

/// Radau IIA methods
pub struct Radau;

use crate::tolerance::Tolerance;
use crate::traits::Real;

/// Trait to allow configuring tolerances on numerical methods generically.
pub trait ToleranceConfig<T: Real> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self;
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self;
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize>
    ToleranceConfig<T> for crate::methods::ExplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize>
    ToleranceConfig<T> for crate::methods::ImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize>
    ToleranceConfig<T> for crate::methods::DiagonallyImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}
