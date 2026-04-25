//! # Differential Algebraic Equations (DAE) Module
//!
//! This module provides comprehensive functionality for solving Differential Algebraic Equations (DAEs),
//! with special focus on Initial Value Problems.
//!
//! ## Example
//!
//! The following example demonstrates how to solve a simple DAE system using the mass matrix formulation:
//!
//! ## Core Components
//!
//! - [`DAE`]: Define your differential algebraic equation system by implementing this trait
//! - [`crate::ivp::Ivp::dae`]: Set up an initial value problem with your system,
//!   time span, and initial conditions
//!

mod dae;
mod numerical_method;
mod solve;

pub use dae::DAE;
pub use numerical_method::AlgebraicNumericalMethod;
pub use solve::solve_dae;
