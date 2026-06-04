//! Ordinary Differential Equations (ODE) module.
//!
//! Provides traits and types for defining and solving ODE initial and boundary value problems.
//! Use [`crate::ivp::IVP::ode`] for the high-level builder API.

mod hamiltonian;
mod numerical_method;
mod ode;
mod solve_bvp;
mod solve_ivp;

pub use hamiltonian::{Hamiltonian, HamiltonianFnWrapper, HamiltonianSystem};
pub use numerical_method::OrdinaryNumericalMethod;
pub use ode::ODE;
pub use solve_bvp::solve_bvp;
pub use solve_ivp::solve_ode;
