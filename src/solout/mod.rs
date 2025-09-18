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
    control::ControlFlag,
    interpolate::Interpolation,
    solution::Solution,
    traits::{Real, State},
};

mod crossing;
mod default;
mod dense;
mod even;
mod event;
mod hyperplane;
mod solout;
mod t_eval;

pub use crossing::CrossingSolout;
pub use default::DefaultSolout;
pub use dense::DenseSolout;
pub use even::EvenSolout;
pub use event::{Event, EventConfig, EventSolout, EventWrappedSolout};
pub use hyperplane::HyperplaneCrossingSolout;
pub use solout::Solout;
pub use t_eval::TEvalSolout;

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

impl From<i8> for CrossingDirection {
    fn from(value: i8) -> Self {
        match value {
            1 => CrossingDirection::Positive,
            -1 => CrossingDirection::Negative,
            _ => CrossingDirection::Both,
        }
    }
}
