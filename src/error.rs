//! NumericalMethod Trait for Differential Equation NumericalMethods

use crate::traits::Real;
use nalgebra::SMatrix;
use std::fmt::{Debug, Display};

/// Error for Differential Equation NumericalMethods
///
/// # Variants
/// * `BadInput` - NumericalMethod input was bad.
/// * `MaxSteps` - NumericalMethod reached maximum steps.
/// * `StepSize` - NumericalMethod terminated due to step size converging too small of a value.
/// * `Stiffness` - NumericalMethod terminated due to stiffness.
///
#[derive(PartialEq, Clone)]
pub enum Error<T, const R: usize, const C: usize>
where
    T: Real,
{
    /// NumericalMethod input was bad
    BadInput {
        msg: String,        // if input is bad, return this with reason
    },
    MaxSteps {              // If the solver reaches the maximum number of steps
        t: T,               // Time at which the solver reached maximum steps
        y: SMatrix<T, R, C>,// Solution at time t
    }, 
    StepSize {
        t: T,               // Time at which step size became too small
        y: SMatrix<T, R, C>,// Solution at time t
    },
    Stiffness {
        t: T,               // Time at which stiffness was detected
        y: SMatrix<T, R, C>,// Solution at time t
    },
}

impl<T, const R: usize, const C: usize> Display for Error<T, R, C>
where
    T: Real + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadInput { msg } => write!(f, "Bad Input: {}", msg),
            Self::MaxSteps { t, y } => write!(f, "Maximum steps reached at (t, y) = ({}, {})", t, y),
            Self::StepSize { t, y } => write!(f, "Step size too small at (t, y) = ({}, {})", t, y),
            Self::Stiffness { t, y } => write!(f, "Stiffness detected at (t, y) = ({}, {})", t, y),
        }
    }
}

impl<T, const R: usize, const C: usize> Debug for Error<T, R, C>
where
    T: Real + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadInput { msg } => write!(f, "Bad Input: {}", msg),
            Self::MaxSteps { t, y } => write!(f, "Maximum steps reached at (t, y) = ({}, {})", t, y),
            Self::StepSize { t, y } => write!(f, "Step size too small at (t, y) = ({}, {})", t, y),
            Self::Stiffness { t, y } => write!(f, "Stiffness detected at (t, y) = ({}, {})", t, y),
        }
    }
}

impl<T, const R: usize, const C: usize> std::error::Error for Error<T, R, C>
where
    T: Real + Debug,
{}
