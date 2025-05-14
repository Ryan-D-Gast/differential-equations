//! Control flow for solving process. Control flags and data return from events, solout functions, etc.

use crate::traits::{Real, State, CallBackData};

/// Control flag for solver execution flow
///
/// ControlFlag is a command to the solver about how to proceed with integration.
/// Used by both event functions and solout functions to control solver execution.
///
#[derive(Debug, Clone)]
pub enum ControlFlag<T = f64, V = f64, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Continue to next step
    Continue,
    /// Modify State and continue to next step
    ModifyState(T, V),
    /// Terminate solver with the given reason/data
    Terminate(D),
}