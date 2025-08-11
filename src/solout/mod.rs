//! Solout trait and common implementations for controlling the output of differential equation solvers.
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
    ControlFlag, Solution,
    interpolate::Interpolation,
    traits::{CallBackData, Real, State},
};

mod solout;
mod default;
mod even;
mod dense;
mod t_eval;
mod crossing;
mod hyperplane;

pub use solout::Solout;
pub use default::DefaultSolout;
pub use even::EvenSolout;
pub use dense::DenseSolout;
pub use t_eval::TEvalSolout;
pub use crossing::CrossingSolout;
pub use hyperplane::HyperplaneCrossingSolout;

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
