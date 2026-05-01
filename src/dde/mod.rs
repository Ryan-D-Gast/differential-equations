//! Delay Differential Equations (DDE) module.
//!
//! Provides traits and types for defining and solving DDE initial value problems.
//! Use [`crate::ivp::IVP::dde`] for the high-level builder API.

mod dde;
mod numerical_method;
mod solve;

pub use dde::DDE;
pub use numerical_method::DelayNumericalMethod;
pub use solve::solve_dde;
