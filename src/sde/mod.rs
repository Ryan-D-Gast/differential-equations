//! Stochastic Differential Equations (SDE) module.
//!
//! Provides traits and types for defining and solving SDE initial value problems.
//! See [`SDE`] and [`SDEProblem`] for usage.

mod sde;
mod numerical_method;
mod problem;
mod solve;

pub use sde::SDE;
pub use numerical_method::StochasticNumericalMethod;
pub use problem::SDEProblem;
pub use solve::solve_sde;
