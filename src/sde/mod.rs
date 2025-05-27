//! # Stochastic Differential Equations (SDE) Module
//!
//! This module provides comprehensive functionality for solving Stochastic Differential Equations (SDEs),
//! focusing on Initial Value Problems (SDEProblems).
//!
//! ## Example
//!
//! The following example demonstrates how to solve a simple Geometric Brownian Motion SDE using the Euler-Maruyama method:
//!
//! ```rust
//! use differential_equations::prelude::*;
//! use nalgebra::SVector;
//! use rand::SeedableRng;
//! use rand_distr::{Distribution, Normal};
//!
//! // Define the SDE system: dY = mu*Y dt + sigma*Y dW
//! struct GBM {
//!     mu: f64,
//!     sigma: f64,
//!     rng: rand::rngs::StdRng,
//! }
//!
//! impl GBM {
//!     fn new(mu: f64, sigma: f64, seed: u64) -> Self {
//!         Self {
//!             mu,
//!             sigma,
//!             rng: rand::rngs::StdRng::seed_from_u64(seed),
//!         }
//!     }
//! }
//!
//! impl SDE<f64, SVector<f64, 1>> for GBM {
//!     fn drift(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
//!         dydt[0] = self.mu * y[0];
//!     }
//!     fn diffusion(&self, _t: f64, y: &SVector<f64, 1>, dydw: &mut SVector<f64, 1>) {
//!         dydw[0] = self.sigma * y[0];
//!     }
//!     fn noise(&self, dt: f64, dw: &mut SVector<f64, 1>) {
//!         let normal = Normal::new(0.0, dt.sqrt()).unwrap();
//!         dw[0] = normal.sample(&mut self.rng.clone());
//!     }
//! }
//!
//! fn main() {
//!     let t0 = 0.0;
//!     let tf = 1.0;
//!     let y0 = SVector::<f64, 1>::new(100.0);
//!     let mu = 0.1;
//!     let sigma = 0.3;
//!     let seed = 42;
//!
//!     let gbm = GBM::new(mu, sigma, seed);
//!     let mut solver = EM::new(0.01);
//!     let problem = SDEProblem::new(gbm, t0, tf, y0);
//!
//!     let solution = match problem.solve(&mut solver) {
//!         Ok(sol) => sol,
//!         Err(e) => panic!("Error: {:?}", e),
//!     };
//!
//!     for (t, y) in solution.iter() {
//!         println!("t: {:.4}, y: {:.4}", t, y[0]);
//!     }
//! }
//! ```
//!
//! ## Core Components
//!
//! - [`SDE`]: Define your stochastic differential equation system by implementing this trait
//! - [`SDEProblem`]: Set up an initial value problem with your system, time span, and initial conditions
//!
//! ## Popular Numerical Methods
//!
//! - [`EM`]: Euler-Maruyama method for SDEs
//! - [`Milstein`]: Milstein method for SDEs
//! - [`RKM4`]: Runge-Kutta 4 method for SDEs
//!

mod sde;
pub use sde::SDE;

pub mod methods;

mod numerical_method;
pub use numerical_method::SDENumericalMethod;

mod problem;
mod solve;

pub use problem::SDEProblem;
pub use solve::solve_sde;
