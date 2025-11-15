//! Status for solving differential equations

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fmt::{Debug, Display};

use crate::{
    error::Error,
    traits::{Real, State},
};

/// Status for solving differential equations
///
/// # Variants
/// * `Uninitialized` - NumericalMethod has not been initialized.
/// * `Initialized`   - NumericalMethod has been initialized.
/// * `Error`         - NumericalMethod encountered an error.
/// * `Solving`       - NumericalMethod is solving.
/// * `RejectedStep`  - NumericalMethod rejected step.
/// * `Interrupted`   - NumericalMethod was interrupted by Solout with reason.
/// * `Complete`      - NumericalMethod completed.
///
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Clone)]
pub enum Status<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// Uninitialized state
    Uninitialized,
    /// Initialized state
    Initialized,
    /// General Error pass through from solver
    Error(Error<T, Y>),
    /// Currently being solved
    Solving,
    /// Step was rejected, typically will retry with smaller step size
    RejectedStep,
    /// Solver was interrupted by Solout function with reason
    Interrupted,
    /// Solver has completed the integration successfully.
    Complete,
}

impl<T, Y> Display for Status<T, Y>
where
    T: Real + Display,
    Y: State<T> + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "NumericalMethod: Uninitialized"),
            Self::Initialized => write!(f, "NumericalMethod: Initialized"),
            Self::Error(err) => write!(f, "NumericalMethod Error: {}", err),
            Self::Solving => write!(f, "NumericalMethod: Solving in progress"),
            Self::RejectedStep => write!(f, "NumericalMethod: Step rejected"),
            Self::Interrupted => write!(f, "NumericalMethod: Interrupted"),
            Self::Complete => write!(f, "NumericalMethod: Complete"),
        }
    }
}
