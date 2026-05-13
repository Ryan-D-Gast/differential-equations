//! Stochastic Differential Equations (SDE) module.
//!
//! Provides traits and types for defining and solving SDE initial value problems.
//! Use [`crate::ivp::IVP::sde`] for the high-level builder API.

mod numerical_method;
mod sde;
mod solve_ivp;

pub use numerical_method::StochasticNumericalMethod;
pub use sde::SDE;
pub use solve_ivp::solve_sde;
