//! # differential-equations
//!
//! A Rust library for solving ODE, DDE, and SDE initial value problems.
//!
//! [![GitHub](https://img.shields.io/badge/GitHub-differential--equations-blue)](https://github.com/Ryan-D-Gast/differential-equations)
//! [![Documentation](https://docs.rs/differential-equations/badge.svg)](https://docs.rs/differential-equations)
//!
//! ## Overview
//!
//! This library provides numerical solvers for:
//!
//! - **[Ordinary Differential Equations (ODE)](crate::ode)**: initial value problems, fixed/adaptive step, event detection, flexible output
//! - **[Differential Algebraic Equations (DAE)](crate::dae)**: equations in the form M f' = f(t,y) where M can be singular
//! - **[Delay Differential Equations (DDE)](crate::dde)**: constant/state-dependent delays, same features as ODE
//! - **[Stochastic Differential Equations (SDE)](crate::sde)**: drift-diffusion, user RNG, same features as ODE
//!
//! ## Feature Flags
//!
//! - `polars`: Enable converting `Solution` to a Polars DataFrame with `Solution::to_polars()`
//!
//! ## Example (ODE)
//!
//! ```rust
//! use differential_equations::prelude::*;
//! use nalgebra::{SVector, vector};
//!
//! pub struct LinearEquation {
//!     pub a: f64,
//!     pub b: f64,
//! }
//!
//! impl ODE<f64, SVector<f64, 1>> for LinearEquation {
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
//!     let problem = ODEProblem::new(system, t0, tf, y0);
//!     let mut solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-6);
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
//! ## License
//!
//! ```text
//! Copyright 2025 Ryan D. Gast
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! ```

// Prelude & User-Facing API
pub mod prelude;

// Numerical Methods
pub mod methods;
pub mod tableau;

// Differential Equations
pub mod dae;
pub mod dde;
pub mod ode;
pub mod sde;

// Output Control
pub mod solout;

// Core Structures
pub mod control;
pub mod error;
pub mod solution;
pub mod stats;
pub mod status;

// Shared Traits & Utilities
pub mod interpolate;
pub mod linalg;
pub mod traits;
pub mod utils;

// Derive Macros
pub mod derive {
    pub use differential_equations_derive::State;
}
