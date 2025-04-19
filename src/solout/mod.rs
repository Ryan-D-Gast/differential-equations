//! Solout trait and common implementations for controlling the output of ODE solvers in the `ode` module.
//!
//! ## Includes
//! * `DefaultSolout` for capturing all solver steps
//! * `EvenSolout` for capturing evenly spaced solution points
//! * `DenseSolout` for capturing a dense set of interpolated points
//! * `TEvalSolout` for capturing points based on a user-defined function
//! * `CrossingSolout` for capturing points when crossing a specified value
//! * `HyperplaneCrossingSolout` for capturing points when crossing a hyperplane
//!

use crate::{
    Solution, ControlFlag,
    traits::{Real, State, CallBackData},
    interpolate::Interpolation,
};

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
