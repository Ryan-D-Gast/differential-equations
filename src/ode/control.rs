//! Control flow for solving process. Control flags and data return from events, solout functions, etc.

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

/// Callback data trait
///
/// This trait represents data that can be returned from functions
/// that are used to control the solver's execution flow. The
/// Clone and Debug traits are required for internal use but anything
/// that implements this trait can be used as callback data.
/// For example, this can be a string, a number, or any other type
/// that implements the Clone and Debug traits.
///
pub trait CallBackData: Clone + std::fmt::Debug {}

// Implement for any type that already satisfies the bounds
impl<T: Clone + std::fmt::Debug> CallBackData for T {}