//! Numerical methods for ODEs.

use crate::{
    error::Error,
    ode::ODE,
    stats::Evals,
    status::Status,
    traits::{Real, State},
};

/// Trait for ODE solvers.
///
/// Implemented by types that can solve ODEs via repeated calls to `step`.
pub trait OrdinaryNumericalMethod<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// Initialize the solver before integration
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y0`     - Initial state.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok if initialization is successful,
    ///
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>;

    /// Advance the solution by one step
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    ///
    /// # Returns
    /// * Result<Evals, Errors<T, Y>> - Ok if step is successful with the number of function evaluations,
    ///
    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>;

    // Accessors

    /// Time of last accepted step
    fn t(&self) -> T;

    /// State at last accepted step
    fn y(&self) -> &Y;

    /// Time of previous accepted step
    fn t_prev(&self) -> T;

    /// State at previous accepted step
    fn y_prev(&self) -> &Y;

    /// Step size for next step
    fn h(&self) -> T;

    /// Set step size for next step
    fn set_h(&mut self, h: T);

    /// Current solver status
    fn status(&self) -> &Status<T, Y>;

    /// Set solver status
    fn set_status(&mut self, status: Status<T, Y>);
}
