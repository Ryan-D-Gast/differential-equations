//! Delay Differential Equations (DDE) module.
//!
//! Provides traits and types for defining and solving DDE initial value problems.
//! See [`DDE`] and [`DDEProblem`] for usage.

mod dde;
mod numerical_method;
mod problem;
mod solve;

pub use dde::DDE;
pub use numerical_method::DelayNumericalMethod;
pub use problem::DDEProblem;
pub use solve::solve_dde;
