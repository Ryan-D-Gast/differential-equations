//! Ordinary Differential Equations (ODE) module.
//!
//! Provides traits and types for defining and solving ODE initial value problems.
//! See [`ODE`] and [`ODEProblem`] for usage.

mod numerical_method;
mod ode;
mod problem;
mod solve;

pub use numerical_method::OrdinaryNumericalMethod;
pub use ode::ODE;
pub use problem::ODEProblem;
pub use solve::solve_ode;
