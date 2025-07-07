//! # Ordinary Differential Equations (ODE) Module
//!
//! This module provides comprehensive functionality for solving Ordinary Differential Equations (ODEs),
//! with special focus on Initial Value Problems (ODEProblems).
//!
//! ## Example
//!
//! The following example demonstrates how to solve a simple linear ODE using the Dormand-Prince 8(5,3) method:
//!
//! ```rust
//! use differential_equations::prelude::*;
//!
//! pub struct LinearEquation {
//!     pub a: f64,
//!     pub b: f64,
//! }
//!
//! // defaults to <Type = f64, State = f64> so can be omitted in this case
//! // Note that the State can be a nalgebra vector of any size,
//! // e.g. SVector<f64, 2> for 2D systems
//! impl ODE for LinearEquation {
//!     fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
//!         *dydt = self.a + self.b * y;
//!     }
//! }
//!
//! fn main() {
//!     // Define the ODE system and Initial Conditions
//!     let system = LinearEquation { a: 1.0, b: 2.0 };
//!     let t0 = 0.0;
//!     let tf = 1.0;
//!     let y0 = 1.0;
//!
//!     // Create an ODEProblem instance
//!     let problem = ODEProblem::new(system, t0, tf, y0);
//!
//!     // Initialize solver with desired settings or use defaults
//!     let mut solver = ExplicitRungeKutta::dop853()
//!         .rtol(1e-8)  // Relative tolerance
//!         .atol(1e-6); // Absolute tolerance
//!
//!     // Solve the ODEProblem
//!     let solution = match problem.solve(&mut solver) {
//!         Ok(sol) => sol,
//!         Err(e) => panic!("Error: {:?}", e),
//!     };
//!
//!     // Print the solution
//!     for (t, y) in solution.iter() {
//!       println!("t: {:.4}, y: {:.4}", t, y);
//!     }
//! }
//! ```
//!
//! ## Core Components
//!
//! - [`ODE`]: Define your differential equation system by implementing this trait
//! - [`ODEProblem`]: Set up an initial value problem with your system, time span, and initial conditions
//!

// Definitions & Constructors for users to ergonomically solve an ODEProblem problem via the solve_ode function.
mod problem;
pub use problem::ODEProblem;

// Solve ODE function
mod solve;
pub use solve::solve_ode;

// ODE Trait for Ordinary Differential Equations
mod ode;
pub use ode::{
    ODE, // ODE Trait for Differential Equations
};

// OrdinaryNumericalMethod Traits for ODE NumericalMethods.
mod numerical_method;
pub use numerical_method::OrdinaryNumericalMethod;