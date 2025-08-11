//! Control flow flags used by solvers, events, and output callbacks.

use crate::traits::{CallBackData, Real, State};

/// Directive to the solver indicating how to proceed with integration.
///
/// Returned by event functions and `Solout` callbacks to steer execution.
///
#[derive(Debug, Clone)]
pub enum ControlFlag<T = f64, Y = f64, D = String>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    /// Continue to the next step.
    Continue,
    /// Replace the current state and continue to the next step.
    ModifyState(T, Y),
    /// Terminate the solver with the given reason/data.
    Terminate(D),
}
