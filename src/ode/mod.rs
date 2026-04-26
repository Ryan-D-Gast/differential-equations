//! Ordinary Differential Equations (ODE) module.
//!
//! Provides traits and types for defining and solving ODE initial value problems.
//! Use [`crate::ivp::Ivp::ode`] for the high-level builder API.

mod numerical_method;
mod ode;
mod sensitivity;
mod solve;

pub use numerical_method::OrdinaryNumericalMethod;
pub use ode::ODE;
pub use sensitivity::{
    AdjointCost, AdjointSolution, AdjointState, ForwardSensitivityODE, VaryParameters,
    solve_adjoint_sensitivity,
};
pub use solve::solve_ode;
