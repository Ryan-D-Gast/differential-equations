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
        Self::BadInput { msg } => write!(f, "Bad input: {}. Check your problem definition, dimensions, and solver options.", msg),
            Self::MaxSteps { t, y } => {
        write!(f, "Maximum step count reached at (t, y) = ({}, {}). Try increasing max_steps, relaxing tolerances, or shortening the integration interval.", t, y)
            }
        Self::StepSize { t, y } => write!(f, "Step size became too small at (t, y) = ({}, {}). This often indicates stiffness or overly tight tolerances. Consider using a stiff solver (DIRK/IRK), relaxing rtol/atol, or rescaling the problem.", t, y),
        Self::Stiffness { t, y } => write!(f, "Stiffness detected at (t, y) = ({}, {}). Switch to a stiff method (e.g., DIRK or IRK) or relax tolerances to improve stability.", t, y),
            Self::OutOfBounds {
                t_interp,
                t_prev,
                t_curr,
            } => {
                write!(
                    f,
            "Interpolation error: requested t_interp {} is outside the last step: [t_prev {}, t_curr {}]. Dense output is only valid within a completed step; request t within this interval or use t_eval to sample during integration.",
            t_interp, t_prev, t_curr
                )
            }
        Self::NoLags => write!(f, "Invalid DDE configuration: number of lags L must be > 0. If there are no delays, use an ODE solver instead."),
            Self::InsufficientHistory { t_delayed, t_prev, t_curr } => {
                write!(
                    f,
            "Insufficient history to interpolate at delayed time {} (window: [t_prev {}, t_curr {}]). Possible causes: max_delay is too small or history pruning is too aggressive, or steps advanced before enough history accumulated. Consider increasing max_delay, providing a longer initial history, or reducing maximum step size.",
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
            Self::BadInput { msg } => write!(f, "Bad input: {}. Check your problem definition, dimensions, and solver options.", msg),
            Self::MaxSteps { t, y } => {
                write!(f, "Maximum step count reached at (t, y) = ({:?}, {:?}). Try increasing max_steps, relaxing tolerances, or shortening the integration interval.", t, y)
            }
            Self::StepSize { t, y } => {
                write!(f, "Step size became too small at (t, y) = ({:?}, {:?}). This often indicates stiffness or overly tight tolerances. Consider using a stiff solver (DIRK/IRK), relaxing rtol/atol, or rescaling the problem.", t, y)
            }
            Self::Stiffness { t, y } => {
                write!(f, "Stiffness detected at (t, y) = ({:?}, {:?}). Switch to a stiff method (e.g., DIRK or IRK) or relax tolerances to improve stability.", t, y)
            }
            Self::OutOfBounds {
                t_interp,
                t_prev,
                t_curr,
            } => {
                write!(
                    f,
                    "Interpolation error: requested t_interp {:?} is outside the last step: [t_prev {:?}, t_curr {:?}]. Dense output is only valid within a completed step; request t within this interval or use t_eval to sample during integration.",
                    t_interp, t_prev, t_curr
                )
            }
            Self::NoLags => write!(f, "Invalid DDE configuration: number of lags L must be > 0. If there are no delays, use an ODE solver instead."),
            Self::InsufficientHistory { t_delayed, t_prev, t_curr } => {
                write!(
                    f,
                    "Insufficient history to interpolate at delayed time {:?} (window: [t_prev {:?}, t_curr {:?}]). Possible causes: max_delay is too small or history pruning is too aggressive, or steps advanced before enough history accumulated. Consider increasing max_delay, providing a longer initial history, or reducing maximum step size.",
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
