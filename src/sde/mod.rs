//! Stochastic Differential Equations (SDE) module.
//!
//! Provides traits and types for defining and solving SDE initial value problems.
//! Use [`crate::ivp::Ivp::sde`] for the high-level builder API.

mod numerical_method;
mod sde;
mod solve;

pub use numerical_method::StochasticNumericalMethod;
pub use sde::SDE;
pub use solve::solve_sde;
