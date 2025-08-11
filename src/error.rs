//! NumericalMethod Trait for Differential Equations Crate

use crate::traits::{Real, State};
use std::fmt::{Debug, Display};

/// Error for Differential Equations Crate
#[derive(PartialEq, Clone)]
pub enum Error<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// NumericalMethod input was bad
    BadInput {
        msg: String, // if input is bad, return this with reason
    },

    /// Maximum steps reached
    MaxSteps {
        t: T,
        y: Y,
    },

    /// Step size became too small
    StepSize {
        t: T,
        y: Y,
    },

    /// Stiffness detected
    Stiffness {
        t: T,
        y: Y,
    },

    /// Out of bounds error
    OutOfBounds {
        t_interp: T,
        t_prev: T,
        t_curr: T,
    },

    /// DDE requires at least one lag (L > 0)
    NoLags,

    /// Not enough history retained to evaluate a delayed state
    InsufficientHistory {
        t_delayed: T,
        t_prev: T,
        t_curr: T,
    },

}

impl<T, Y> Display for Error<T, Y>
where
    T: Real + Display,
    Y: State<T> + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadInput { msg } => write!(f, "Bad Input: {}", msg),
            Self::MaxSteps { t, y } => {
                write!(f, "Maximum steps reached at (t, y) = ({}, {})", t, y)
            }
            Self::StepSize { t, y } => write!(f, "Step size too small at (t, y) = ({}, {})", t, y),
            Self::Stiffness { t, y } => write!(f, "Stiffness detected at (t, y) = ({}, {})", t, y),
            Self::OutOfBounds {
                t_interp,
                t_prev,
                t_curr,
            } => {
                write!(
                    f,
                    "Interpolation Error: t_interp {} is not within the previous and current step: t_prev {}, t_curr {}",
                    t_interp, t_prev, t_curr
                )
            }
            Self::NoLags => write!(f, "Invalid DDE configuration: number of lags L must be > 0"),
            Self::InsufficientHistory { t_delayed, t_prev, t_curr } => {
                write!(
                    f,
                    "Insufficient history to interpolate at t_delayed {} (t_prev {}, t_curr {})",
                    t_delayed, t_prev, t_curr
                )
            }
        }
    }
}

impl<T, Y> Debug for Error<T, Y>
where
    T: Real + Debug,
    Y: State<T> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadInput { msg } => write!(f, "Bad Input: {}", msg),
            Self::MaxSteps { t, y } => {
                write!(f, "Maximum steps reached at (t, y) = ({:?}, {:?})", t, y)
            }
            Self::StepSize { t, y } => {
                write!(f, "Step size too small at (t, y) = ({:?}, {:?})", t, y)
            }
            Self::Stiffness { t, y } => {
                write!(f, "Stiffness detected at (t, y) = ({:?}, {:?})", t, y)
            }
            Self::OutOfBounds {
                t_interp,
                t_prev,
                t_curr,
            } => {
                write!(
                    f,
                    "Interpolation Error: t_interp {:?} is not within the previous and current step: t_prev {:?}, t_curr {:?}",
                    t_interp, t_prev, t_curr
                )
            }
            Self::NoLags => write!(f, "Invalid DDE configuration: number of lags L must be > 0"),
            Self::InsufficientHistory { t_delayed, t_prev, t_curr } => {
                write!(
                    f,
                    "Insufficient history to interpolate at t_delayed {:?} (t_prev {:?}, t_curr {:?})",
                    t_delayed, t_prev, t_curr
                )
            }
        }
    }
}

impl<T, Y> std::error::Error for Error<T, Y>
where
    T: Real + Debug + Display,
    Y: State<T> + Debug + Display,
{
}
