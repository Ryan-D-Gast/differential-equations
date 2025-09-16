//! Status for solving differential equations

use std::fmt::{Debug, Display};

use crate::{
    error::Error,
    traits::{CallBackData, Real, State},
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
#[derive(Debug, PartialEq, Clone)]
pub enum Status<T, Y, D>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    /// Uninitialized state
    Uninitialized,
    /// Initialized state
    Error(Error<T, Y>),
    /// Currently being solved
    Solving,
    /// Step was rejected, typically will retry with smaller step size
    RejectedStep,
    /// Solver was interrupted by Solout function with reason
    Interrupted(D),
    /// Solver has completed the integration successfully.
    Complete,
}

impl<T, Y, D> Display for Status<T, Y, D>
where
    T: Real + Display,
    Y: State<T> + Display,
    D: CallBackData + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "NumericalMethod: Uninitialized"),
            Self::Initialized => write!(f, "NumericalMethod: Initialized"),
            Self::Error(err) => write!(f, "NumericalMethod Error: {}", err),
            Self::Solving => write!(f, "NumericalMethod: Solving in progress"),
            Self::RejectedStep => write!(f, "NumericalMethod: Step rejected"),
            Self::Interrupted(reason) => write!(f, "NumericalMethod: Interrupted - {}", reason),
            Self::Complete => write!(f, "NumericalMethod: Complete"),
        }
    }
}
