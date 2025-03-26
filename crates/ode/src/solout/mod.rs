//! ode module that exports most commonly used items
//!
//! This module provides the most commonly used types, traits, and functions
//! from differential-equations so they can be imported with a single `use` statement:
//!
//! ```
//! use differential_equations::ode::*;
//! ```
//! 
//! # Includes
//! 
//! ## Core API
//! * `IVP` struct for defining initial value problems
//! * `Solution` struct for storing the results of a solved IVP
//! * `SolverStatus` enum for checking the status of the solver
//! 
//! ## Defining systems and solout
//! * `System` trait for defining system of differential equations
//! * `EventAction` return enum for system.event function
//! * `Solout` trait for controlling output of the solver
//! 
//! ## Solvers
//! * `RK4` Classic 4th-order Runge-Kutta method
//! * `DOP853` Dormand-Prince 8th-order method
//! * `DOPRI5` Dormand-Prince 5th-order method
//! * `APCF4` Fixed step 4th-order Adams Predictor-Corrector method
//! * `APCV4` Adaptive step 4th-order Adams Predictor-Corrector method
//! 
//! Note more solvers are available in the `solvers` module.
//! 
//! ## Solout
//! * `DefaultSolout` for capturing all solver steps
//! * `EvenSolout` for capturing evenly spaced solution points
//! * `DenseSolout` for capturing a dense set of interpolated points
//! * `TEvalSolout` for capturing points based on a user-defined function
//! * `CrossingSolout` for capturing points when crossing a specified value
//! * `HyperplaneCrossingSolout` for capturing points when crossing a hyperplane
//! 
//! Note more solout options are available in the `solout` module.
//! 
//! # Miscellaneous traits to expose API
//! * `Solver` trait for defining solvers, not used by users but here so trait methods can be used.
//! 
//! # License
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
//! 

use nalgebra::SMatrix;
use crate::{Solver, System, EventData};
use crate::traits::Real;
use crate::interpolate::find_cubic_hermite_crossing;

// Solout Trait for controlling output of the solver
mod solout;
pub use solout::Solout;

// Common Solout Implementations
mod default;
pub use default::DefaultSolout;

mod even;
pub use even::EvenSolout;

mod dense;
pub use dense::DenseSolout;

mod t_eval;
pub use t_eval::TEvalSolout;

// Crossing Detecting Solouts 

/// Defines the direction of threshold crossing to detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingDirection {
    /// Detect crossings in both directions
    Both,
    /// Detect only crossings from below to above the threshold (positive direction)
    Positive,
    /// Detect only crossings from above to below the threshold (negative direction)
    Negative,
}

// Crossing detection solout
mod crossing;
pub use crossing::CrossingSolout;

// Hyperplane crossing detection solout
mod hyperplane;
pub use hyperplane::HyperplaneCrossingSolout;