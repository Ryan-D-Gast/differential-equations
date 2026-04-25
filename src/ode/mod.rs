//! Ordinary Differential Equations (ODE) module.
//!
//! Provides traits and types for defining and solving ODE initial value problems.
//! Use [`crate::ivp::Ivp::ode`] for the high-level builder API.

mod fsa;
mod numerical_method;
mod ode;
mod solve;

pub use fsa::ForwardSensitivityProblem;
pub use numerical_method::OrdinaryNumericalMethod;
pub use ode::ODE;
pub use solve::solve_ode;
