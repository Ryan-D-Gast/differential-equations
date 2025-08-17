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
/// * `Interrupted`   - NumericalMethod was interrupted by event with reason.
/// * `Complete`      - NumericalMethod completed.
///
#[derive(Debug, PartialEq, Clone)]
pub enum Status<T, Y, D>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    Uninitialized,      // NumericalMethods default to this until solver.init is called
    Initialized,        // After solver.init is called
    Error(Error<T, Y>), // If the solver encounters an error, this status is set so solver status is indicated that an error.
    Solving,            // While the Differential Equation is being solved
    RejectedStep, // If the solver rejects a step, in this case it will repeat with new smaller step size typically, will return to Solving once the step is accepted
    Interrupted(D), // If the solver is interrupted by event with reason
    Complete, // If the solver is solving and has reached the final time of the IMatrix<T, R, C, S>P then Complete is returned to indicate such.
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
