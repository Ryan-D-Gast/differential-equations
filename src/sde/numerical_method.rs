//! Numerical methods for SDEs.

use crate::{
    error::Error,
    sde::SDE,
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
};

/// Trait for SDE solvers.
///
/// Implemented by types that can solve SDEs via repeated calls to `step`.
pub trait StochasticNumericalMethod<T, Y, D = String>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    /// Initialize the solver before integration
    ///
    /// # Arguments
    /// * `sde`    - System of SDEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok with number of evaluations if initialization is successful
    ///
    fn init<F>(&mut self, sde: &mut F, t0: T, tf: T, y: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: SDE<T, Y, D>;

    /// Advance the solution by one step
    ///
    /// # Arguments
    /// * `sde`    - System of SDEs to solve.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok with number of evaluations if step is successful
    ///
    fn step<F>(&mut self, sde: &mut F) -> Result<Evals, Error<T, Y>>
    where
        F: SDE<T, Y, D>;

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
    fn status(&self) -> &Status<T, Y, D>;

    /// Set solver status
    fn set_status(&mut self, status: Status<T, Y, D>);
}
