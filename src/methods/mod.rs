//! Numerical Methods for Differential Equations

use crate::{
    tolerance::Tolerance,
    traits::{Real, State},
};

mod h_init;

mod apc;
mod dirk;
mod erk;
mod irk;

pub use apc::AdamsPredictorCorrector;
pub use dirk::DiagonallyImplicitRungeKutta;
pub use erk::ExplicitRungeKutta;
pub use irk::ImplicitRungeKutta;

// Typestate categories for differential equation types.
pub struct Ordinary;
pub struct Delay;
pub struct Stochastic;
pub struct Algebraic;

/// Fixed-step methods
pub struct Fixed;

/// Adaptive-step methods
pub struct Adaptive;

/// Explicit Adaptive-step methods by Dormand-Prince
pub struct DormandPrince;

/// Radau IIA methods
pub struct Radau;

/// Trait to allow configuring tolerances on numerical methods generically.
pub trait ToleranceConfig<T: Real> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self;
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self;
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for crate::methods::ExplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for crate::methods::ImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for crate::methods::DiagonallyImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}
