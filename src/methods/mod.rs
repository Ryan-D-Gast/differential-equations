//! Numerical Methods for Differential Equations

use crate::{
    tolerance::Tolerance,
    traits::{Real, State},
};

mod h_init;

mod apc;
mod bdf;
mod dirk;
mod erk;
mod irk;
mod milstein;

pub mod bvp;

pub use apc::AdamsPredictorCorrector;
pub use bdf::BackwardDifferentiationFormula;
pub use dirk::DiagonallyImplicitRungeKutta;
pub use erk::ExplicitRungeKutta;
pub use irk::ImplicitRungeKutta;
pub use milstein::Milstein;

// Typestate categories for differential equation types.
#[derive(Clone)]
pub struct Ordinary;
#[derive(Clone)]
pub struct Delay;
#[derive(Clone)]
pub struct Stochastic;
#[derive(Clone)]
pub struct Algebraic;

/// Fixed-step methods
#[derive(Clone)]
pub struct Fixed;

/// Adaptive-step methods
#[derive(Clone)]
pub struct Adaptive;

/// Explicit Adaptive-step methods by Dormand-Prince
#[derive(Clone)]
pub struct DormandPrince;

/// Radau IIA methods
#[derive(Clone)]
pub struct Radau;

/// Trait to allow configuring tolerances on numerical methods generically.
pub trait ToleranceConfig<T: Real> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self;
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self;
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for ExplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for ImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T>
    for DiagonallyImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}
