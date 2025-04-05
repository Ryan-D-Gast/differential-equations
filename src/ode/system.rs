//! Defines system of differential equations for numerical solvers.
//! The Solvers use this trait to take a input system from the user and solve
//! Includes a differential equation and optional solout function to interupt solver
//! given a condition or event.

use crate::traits::Real;
use nalgebra::SMatrix;

/// ODE Trait for Differential Equations
/// 
/// ODE trait defines the differential equation dydt = f(t, y) for the solver.
/// The differential equation is used to solve the ordinary differential equation.
/// The trait also includes a solout function to interupt the solver when a condition
/// is met or event occurs.
/// 
/// # Impl
/// * `diff`    - Differential Equation dydt = f(t, y) in form f(t, &y, &mut dydt).
/// * `event`   - Solout function to interupt solver when condition is met or event occurs.
/// 
/// Note that the solout function is optional and can be left out when implementing 
/// in which can it will be set to return false by default.
/// 
#[allow(unused_variables)]
pub trait ODE<T = f64, const R: usize = 1, const C: usize = 1, E = String> 
where 
    T: Real,
    E: EventData
{
    /// Differential Equation dydt = f(t, y)
    /// 
    /// An ordinary differential equation (ODE) takes a independent variable
    /// which in this case is 't' as it is typically time and a dependent variable
    /// which is a vector of values 'y'. The ODE returns the derivative of the
    /// dependent variable 'y' with respect to the independent variable 't' as
    /// dydt = f(t, y).
    /// 
    /// For efficiency and ergonomics the derivative is calculated from an argument 
    /// of a mutable reference to the derivative vector dydt. This allows for a 
    /// derivatives to be calculated in place which is more efficient as iterative 
    /// ODE solvers require the derivative to be calculated at each step without 
    /// regard to the previous value.
    /// 
    /// # Arguments
    /// * `t`    - Independent variable point.
    /// * `y`    - Dependent variable point.
    /// * `dydt` - Derivative point.
    /// 
    fn diff(&self, t: T, y: &SMatrix<T, R, C>, dydt: &mut SMatrix<T, R, C>);

    /// Event function to detect significant conditions during integration.
    /// 
    /// Called after each step to detect events like threshold crossings,
    /// singularities, or other mathematically/physically significant conditions.
    /// Can be used to terminate integration when conditions occur.
    /// 
    /// # Arguments
    /// * `t`    - Current independent variable point.
    /// * `y`    - Current dependent variable point.
    /// * `dydt` - Current derivative point.
    /// 
    /// # Returns
    /// * `EventAction` - Command to continue or stop solver.
    /// 
    fn event(&self, t: T, y: &SMatrix<T, R, C>) -> EventAction<E> {
        EventAction::Continue
    }
}

/// Termination Condition for ODE Solver
/// 
/// EventAction is a command to the solver to continue or stop the integration.
/// The solver will continue 
/// # Variants
/// * `Continue`    - Continue to next step.
/// * `Terminate`   - Terminate solver with reason.
/// 
pub enum EventAction<E = String> 
where 
    E: EventData { 
    /// Continue to next step
    Continue,
    /// Terminate solver
    Terminate(E),
}

/// Event data type for ODE implementations
///
/// This trait represents data that can be returned from event functions
/// in ODE implementations. It's designed to be flexible while ensuring
/// the minimal requirements needed for event handling.
/// 
pub trait EventData: Clone + std::fmt::Debug {}

// Implement for any type that already satisfies the bounds
impl<T: Clone + std::fmt::Debug> EventData for T {}