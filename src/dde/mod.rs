//! # Delay Differential Equations (DDE) Module
//!
//! This module provides comprehensive functionality for solving Delay Differential Equations (DDEs),
//! with a focus on Initial Value Problems (DDEProblems).
//!
//! ## Example
//!
//! The following example demonstrates how to solve the Mackey-Glass delay differential equation, a classic DDE known for chaotic behavior:
//!
//! ```rust
//! use differential_equations::prelude::*;
//!
//! struct MackeyGlass {
//!     beta: f64,
//!     gamma: f64,
//!     n: f64,
//!     tau: f64,
//! }
//!
//! impl DDE<1> for MackeyGlass {
//!     fn diff(&self, _t: f64, y: &f64, yd: &[f64; 1], dydt: &mut f64) {
//!         *dydt = (self.beta * yd[0]) / (1.0 + yd[0].powf(self.n)) - self.gamma * *y;
//!     }
//!     fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
//!         lags[0] = self.tau;
//!     }
//! }
//!
//! fn main() {
//!     let mut solver = ExplicitRungeKutta::cash_karp().max_delay(20.0);
//!     let dde = MackeyGlass { beta: 0.2, gamma: 0.1, n: 10.0, tau: 20.0 };
//!     let t0 = 0.0;
//!     let tf = 200.0;
//!     let y0 = 0.1;
//!     let phi = |_t: f64| y0;
//!
//!     let problem = DDEProblem::new(dde, t0, tf, y0, phi);
//!     match problem.even(2.0).solve(&mut solver) {
//!         Ok(solution) => {
//!             for (t, y) in solution.iter() {
//!                 println!("({:.4}, {:.4})", t, y);
//!             }
//!         }
//!         Err(e) => panic!("Error solving DDE: {:?}", e),
//!     }
//! }
//! ```
//!
//! ## Core Components
//!
//! - [`DDE`]: Define your delay differential equation system by implementing this trait
//! - [`DDEProblem`]: Set up an initial value problem with your system, time span, initial conditions, and history function
//!

mod problem;
pub use problem::DDEProblem;

mod solve;
pub use solve::solve_dde;

mod dde;
pub use dde::DDE;

mod numerical_method;
pub use numerical_method::DelayNumericalMethod;