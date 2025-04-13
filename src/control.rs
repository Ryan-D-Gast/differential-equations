//! Control flow for solving process. Control flags and data return from events, solout functions, etc.

use crate::traits::CallBackData;

/// Control flag for solver execution flow
///
/// ControlFlag is a command to the solver about how to proceed with integration.
/// Used by both event functions and solout functions to control solver execution.
///
pub enum ControlFlag<D = String>
where
    D: CallBackData,
{
    /// Continue to next step
    Continue,
    /// Terminate solver with the given reason/data
    Terminate(D),
}