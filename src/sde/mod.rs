//! # Stochastic Differential Equations (SDE) Module
//!
//! This module provides comprehensive functionality for solving Stochastic Differential Equations (SDEs),
//! with support for various numerical methods and noise processes.
//!
//! ## Example
//!
//! The following example demonstrates how to solve the Heston stochastic volatility model:
//!
//! ```rust
//! use differential_equations::prelude::*; 
//! use rand::SeedableRng;
//! use rand_distr::Distribution;
//!
//! // Custom state type for our 2D state vector (price and variance)
//! #[derive(State)]
//! struct HestonState<T> {
//!     price: T,      // Asset price S
//!     variance: T,   // Variance (volatility squared) v
//! }
//!
//! #[derive(Clone)]
//! struct HestonModel {
//!     mu: f64,      // Drift of asset price
//!     kappa: f64,   // Mean reversion speed of volatility
//!     theta: f64,   // Long-term mean of volatility
//!     sigma: f64,   // Volatility of volatility
//!     rho: f64,     // Correlation between price and volatility Wiener processes
//!     rng: rand::rngs::StdRng, // Random number generator
//! }
//!
//! impl HestonModel {
//!     fn new(mu: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, seed: u64) -> Self {
//!         Self {
//!             mu, kappa, theta, sigma, rho,
//!             rng: rand::rngs::StdRng::seed_from_u64(seed),
//!         }
//!     }
//! }
//!
//! impl SDE<f64, HestonState<f64>> for HestonModel {
//!     fn drift(&self, _t: f64, y: &HestonState<f64>, dydt: &mut HestonState<f64>) {
//!         // Asset price drift: μS
//!         dydt.price = self.mu * y.price;
//!         
//!         // Variance drift: κ(θ-v)
//!         dydt.variance = self.kappa * (self.theta - y.variance);
//!     }
//!
//!     fn diffusion(&self, _t: f64, y: &HestonState<f64>, dydw: &mut HestonState<f64>) {
//!         // Asset price diffusion: √v·S
//!         dydw.price = y.price * y.variance.sqrt();
//!         
//!         // Variance diffusion: σ√v
//!         dydw.variance = self.sigma * y.variance.sqrt();
//!     }
//!
//!     fn noise(&self, dt: f64, dw_vec: &mut HestonState<f64>) {
//!         // Generate correlated Wiener process increments
//!         let normal = rand_distr::Normal::new(0.0, dt.sqrt()).unwrap();
//!         
//!         let dw1 = normal.sample(&mut self.rng.clone());
//!         let dw2 = normal.sample(&mut self.rng.clone());
//!         
//!         dw_vec.price = dw1;
//!         dw_vec.variance = self.rho * dw1 + (1.0 - self.rho * self.rho).sqrt() * dw2;
//!     }
//! }
//!
//! fn main() {
//!     // Model parameters
//!     let mu = 0.1;
//!     let kappa = 2.0;
//!     let theta = 0.04;
//!     let sigma = 0.3;
//!     let rho = -0.7;
//!     let seed = 42;
//!     
//!     // Simulation parameters
//!     let t0 = 0.0;
//!     let tf = 1.0;
//!     let dt = 0.01;
//!     
//!     // Initial conditions
//!     let s0 = 100.0;
//!     let v0 = 0.04;
//!     let y0 = HestonState { price: s0, variance: v0 };
//!     
//!     // Create SDE system
//!     let sde = HestonModel::new(mu, kappa, theta, sigma, rho, seed);
//!     
//!     // Create solver with fixed step size
//!     let mut solver = Milstein::new(dt);
//!     
//!     // Create and solve the problem
//!     let problem = SDEProblem::new(sde, t0, tf, y0);
//!     let solution = problem.solve(&mut solver).unwrap();
//!     
//!     // Print final state
//!     let final_price = solution.y.last().unwrap().price;
//!     let final_variance = solution.y.last().unwrap().variance;
//!     println!("Final price: {:.4}, Final volatility: {:.4}%", 
//!              final_price, final_variance.sqrt() * 100.0);
//! }
//! ```
//!
//! ## Core Components
//!
//! - [`SDE`]: Define your stochastic differential equation system by implementing this trait
//! - [`SDEProblem`]: Set up a problem with your system, time span, and initial conditions
//! - [`Solution`]: After solving, analyze and process your solution data
//!
//! ## Popular Numerical Methods
//!
//! - [`Euler`]: Simple Euler-Maruyama method (strong order 0.5)
//! - [`Milstein`]: Milstein method (strong order 1.0)
//! - [`RK4`]: Stochastic Runge-Kutta 4 method
//!
//! Additional solvers are available in the [`methods`] module.
//!
//! ## Advanced Features
//!
//! - Custom state types using the `#[derive(State)]` macro
//! - Support for correlated noise processes
//! - Configurable random number generators
//!

mod sde;
pub use sde::SDE;

pub mod methods;

mod numerical_method;
pub use numerical_method::{
    NumEvals, 
    NumericalMethod
};

mod solve_sde;
mod sde_problem;

pub use solve_sde::solve_sde;
pub use sde_problem::SDEProblem;