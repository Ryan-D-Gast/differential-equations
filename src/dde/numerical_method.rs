//! Numerical methods for DDEs.

use crate::{
    Error, Status,
    dde::DDE,
    stats::Evals,
    traits::{CallBackData, Real, State},
};

/// Trait for DDE solvers.
///
/// Implemented by types that can solve DDEs via repeated calls to `step`.
pub trait DelayNumericalMethod<const L: usize, T, Y, H, D = String>
where
    T: Real,
    Y: State<T>,
    H: Fn(T) -> Y,
    D: CallBackData,
{
    /// Initialize the solver before integration
    ///
    /// # Arguments
    /// * `dde` - System of DDEs to solve.
    /// * `t0`  - Initial time.
    /// * `tf`  - Final time.
    /// * `y0`  - Initial state at `t0`.
    /// * `phi` - Initial history function `phi(t)` returning state `Y` for `t <= t0`.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok(evals) if initialization is successful, Err otherwise.
    ///
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &Y, phi: &H) -> Result<Evals, Error<T, Y>>
    where
        F: DDE<L, T, Y, D>;

    /// Advance the solution by one step
    ///
    /// # Arguments
    /// * `dde`            - System of DDEs to solve.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok(evals) if step is successful, Err otherwise.
    ///
    fn step<F>(&mut self, dde: &F, phi: &H) -> Result<Evals, Error<T, Y>>
    where
        F: DDE<L, T, Y, D>;

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
