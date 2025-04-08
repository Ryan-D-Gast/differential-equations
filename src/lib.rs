//! # differential-equations
//!
//! A Rust library for solving various types of differential equations.
//!
//! [![GitHub](https://img.shields.io/badge/GitHub-differential--equations-blue)](https://github.com/Ryan-D-Gast/differential-equations)
//! [![Documentation](https://docs.rs/differential-equations/badge.svg)](https://docs.rs/differential-equations)
//!
//! ## Overview
//!
//! This library provides numerical solvers for different classes of differential equations:
//!
//! - **[Ordinary Differential Equations (ODE)](crate::ode)**: Stable
//!   - Initial value problems (IVP)
//!   - Fixed and adaptive step methods
//!   - Event detection
//!   - Customizable output control
//!
//! ## Feature Flags
//!
//! - `polars`: Enable converting Solution to Polars DataFrame using `Solution.to_polars()`
//!
//! ## Example (ODE)
//!
//!```rust
//! use differential_equations::ode::*;
//! use nalgebra::{SVector, vector};
//!
//! pub struct LinearEquation {
//!     pub a: f64,
//!     pub b: f64,
//! }
//!
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
//!         Ok(sol) => sol,
//!         Err(e) => panic!("Error: {:?}", e),
//!     };
//!
//!     for (t, y) in solution.iter() {
//!        println!("t: {:.4}, y: {:.4}", t, y[0]);
//!     }
//! }
//!```
//!
//! # License
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

// Re-export of nalgebra types for macros and convenience
pub use nalgebra::{SMatrix, SVector, matrix, vector};

// Traits for Floating Point Types
pub mod traits;

// Interpolation Functions
pub mod interpolate;

// Ordinary Differential Equations (ODE) Module
pub mod ode;
