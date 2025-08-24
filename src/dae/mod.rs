//! # Differential Algebraic Equations (DAE) Module
//!
//! This module provides comprehensive functionality for solving Differential Algebraic Equations (DAEs),
//! with special focus on Initial Value Problems (DAEProblems).
//!
//! ## Example
//!
//! The following example demonstrates how to solve a simple DAE system using the mass matrix formulation:
//!
//! ## Core Components
//!
//! - [`DAE`]: Define your differential algebraic equation system by implementing this trait
//! - [`DAEProblem`]: Set up an initial value problem with your system, time span, and initial conditions
//!

// Definitions & Constructors for users to ergonomically solve a DAEProblem via the solve_dae function.
mod problem;
pub use problem::DAEProblem;

// Solve DAE function
mod solve;
pub use solve::solve_dae;

// DAE Trait for Differential Algebraic Equations
mod dae;
pub use dae::{
    DAE, // DAE Trait for Differential Algebraic Equations
};

// AlgebraicNumericalMethod Traits for DAE NumericalMethods.
mod numerical_method;
pub use numerical_method::AlgebraicNumericalMethod;
