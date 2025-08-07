//! Numerical Methods for solving delay differential equations (DDEs).

use crate::{
    Error, Status,
    dde::DDE,
    stats::Evals,
    traits::{CallBackData, Real, State},
};

/// DelayNumericalMethod Trait for DDE NumericalMethods
///
/// DDE NumericalMethods implement this trait to solve delay differential equations.
/// The `step` function is called iteratively by a solver function (like `solve_dde`)
/// to advance the solution.
///
pub trait DelayNumericalMethod<const L: usize, T, Y, H, D = String>
where
    T: Real,
    Y: State<T>,
    H: Fn(T) -> Y,
    D: CallBackData,
{
    /// Initialize DelayNumericalMethod before solving DDE.
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

    /// Perform one integration step for the DDE.
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

    // Access fields of the solver

    /// Access time of the current state (end of the last accepted step).
    fn t(&self) -> T;

    /// Access solution state `y` at the current time `t`.
    fn y(&self) -> &Y;

    /// Access time at the beginning of the last accepted step.
    fn t_prev(&self) -> T;

    /// Access solution state `y` at the beginning of the last accepted step.
    fn y_prev(&self) -> &Y;

    /// Access the proposed step size `h` for the *next* step attempt.
    fn h(&self) -> T;

    /// Set the step size `h` for the *next* step attempt.
    fn set_h(&mut self, h: T);

    /// Get the current status of the solver (Solving, Complete, Error, etc.).
    fn status(&self) -> &Status<T, Y, D>;

    /// Set the status of the solver.
    fn set_status(&mut self, status: Status<T, Y, D>);
}
