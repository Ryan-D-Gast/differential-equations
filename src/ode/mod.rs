//! Ordinary Differential Equations (ODE) module.
//!
//! Provides traits and types for defining and solving ODE initial value problems.
//! See [`ODE`] and [`ODEProblem`] for usage.

mod problem;
mod solve;
mod ode;
mod numerical_method;

pub use problem::ODEProblem;
pub use solve::solve_ode;
pub use ode::ODE;
pub use numerical_method::OrdinaryNumericalMethod;
