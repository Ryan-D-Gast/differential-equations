//! # Ordinary Differential Equations (ODE) Module
//! 
//! This module provides comprehensive functionality for solving Ordinary Differential Equations (ODEs),
//! with special focus on Initial Value Problems (IVPs).
//! 
//! ## Quick Start
//! 
//! Import the entire module to access all commonly used components:
//! 
//! ```rust
//! use differential_equations::ode::*;
//! ```
//! 
//! ## Example
//! 
//! The following example demonstrates how to solve a simple linear ODE using the Dormand-Prince 8(5,3) method:
//! 
//! ```rust
//! use differential_equations::ode::*;
//! use nalgebra::{SVector, vector};
//! 
//! pub struct LinearEquation {
//!     pub a: f64,
//!     pub b: f64,
//! }
//! 
//! // f64: Time Type, 1: Column Dimension, 1: Row Dimension,
//! // defaults to <f64, 1, 1> so can be omitted in this case
//! impl ODE<f64, 1, 1> for LinearEquation {
//!     fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
//!         dydt[0] = self.a + self.b * y[0];
//!     }
//! }
//! 
//! fn main() {
//!     let system = LinearEquation { a: 1.0, b: 2.0 };
//!     let t0 = 0.0;
//!     let tf = 1.0;
//!     let y0 = vector![1.0];
//!     let ivp = IVP::new(system, t0, tf, y0);
//!     let mut solver = DOP853::new().rtol(1e-8).atol(1e-6);
//!     let solution = match ivp.solve(&mut solver) {
//!     Ok(sol) => sol,
//!     Err(e) => panic!("Error: {:?}", e),
//! };
//! 
//!     for (t, y) in solution.iter() {
//!       println!("t: {:.4}, y: {:.4}", t, y[0]);
//!     }
//! }
//! ```
//! 
//! ## Core Components
//! 
//! - [`ODE`]: Define your differential equation system by implementing this trait
//! - [`IVP`]: Set up an initial value problem with your system, time span, and initial conditions
//! - [`Solution`]: After solving, analyze and process your solution data
//! 
//! ## Available Solvers
//! 
//! The most popular solvers are available at the top level:
//! 
//! - [`RK4`]: Fixed step 4th order Runge-Kutta method
//! - [`DOPRI5`]: Adaptive step Dormand-Prince 5(4) method
//! - [`DOP853`]: Adaptive step Dormand-Prince 8(5,3) method with 7th order interpolant
//! 
//! Additional solvers are available in the [`solvers`] module.
//! 
//! ## Solution Output Control
//! 
//! Custom solution output behaviors can be defined using the [`Solout`] trait.
//! Common implementations are available in the [`solout`] module.
//! 
//! ## Advanced Usage
//! 
//! For more control over the solving process, you can use [`solve_ivp`] directly
//! with custom solvers and output handlers.
//! 
//! ## Event Handling
//! 
//! The [`ODE`] trait includes event handling capabilities for detecting
//! and responding to significant conditions during integration.
//! 

// IVP Struct which is used to solve the ODE given the system and a solver
mod ivp;
pub use ivp::{
    IVP, // Initial Value Problem (IVP) for the system of ODEs
    solve_ivp, // Function to solve the IVP, used internally in IVP Struct
};

// ODE Trait for Differential Equations
mod system;
pub use system::{
    ODE,         // ODE Trait for Differential Equations
    EventAction, // Command to the Solver for Returned in Terminate Function in ODE Trait
    EventData,   // Data returned from the event function in the ODE Trait
};

// Solver Traits for ODE Solvers.
mod solver;
pub use solver::{
    Solver,       // Solver Trait for ODE Solvers
    SolverStatus, // Status of the Solver for Control Flow and Error Handling
};

// Solout Trait for controlling output of the solver
pub mod solout; // Numerous implementations of the Solout trait are contained in this module
pub use solout::{
    // Solout Trait for controlling output of the solver
    Solout,
    CrossingDirection,
};

// Solution of a solved IVP Problem
mod solution;
pub use solution::Solution;

// Solver for ODEs
pub mod solvers;
pub use solvers::{
    // Re-exporting popular solvers to ode module for quick access
    RK4,    // Fixed Step Runge-Kutta 4th Order Solver
    DOPRI5, // Adaptive Step Dormand-Prince 5(4) Solver
    DOP853, // Adaptive Step Dormand-Prince 8(5,3) Solver with 7th order interpolant
};