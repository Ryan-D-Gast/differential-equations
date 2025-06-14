//! Numerical Methods for solving delay differential equations (DDEs).

use crate::{
    Error, Status,
    alias::Evals,
    dde::DDE,
    traits::{CallBackData, Real, State},
};

/// DDENumericalMethod Trait for DDE NumericalMethods
///
/// DDE NumericalMethods implement this trait to solve delay differential equations.
/// The `step` function is called iteratively by a solver function (like `solve_dde`)
/// to advance the solution.
///
pub trait DDENumericalMethod<const L: usize, T, V, H, D = String>
where
    T: Real,
    V: State<T>,
    H: Fn(T) -> V,
    D: CallBackData,
{
    /// Initialize DDENumericalMethod before solving DDE.
    ///
    /// # Arguments
    /// * `dde` - System of DDEs to solve.
    /// * `t0`  - Initial time.
    /// * `tf`  - Final time.
    /// * `y0`  - Initial state at `t0`.
    /// * `phi` - Initial history function `phi(t)` returning state `V` for `t <= t0`.
    ///
    /// # Returns
    /// * Result<NumEvals, Error<T, V>> - Ok(evals) if initialization is successful, Err otherwise.
    ///
    fn init<F>(&mut self, dde: &F, t0: T, tf: T, y0: &V, phi: &H) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>;

    /// Perform one integration step for the DDE.
    ///
    /// # Arguments
    /// * `dde`            - System of DDEs to solve.
    ///
    /// # Returns
    /// * Result<NumEvals, Error<T, V>> - Ok(evals) if step is successful, Err otherwise.
    ///
    fn step<F>(&mut self, dde: &F, phi: &H) -> Result<Evals, Error<T, V>>
    where
        F: DDE<L, T, V, D>;

    // Access fields of the solver

    /// Access time of the current state (end of the last accepted step).
    fn t(&self) -> T;

    /// Access solution state `y` at the current time `t`.
    fn y(&self) -> &V;

    /// Access time at the beginning of the last accepted step.
    fn t_prev(&self) -> T;

    /// Access solution state `y` at the beginning of the last accepted step.
    fn y_prev(&self) -> &V;

    /// Access the proposed step size `h` for the *next* step attempt.
    fn h(&self) -> T;

    /// Set the step size `h` for the *next* step attempt.
    fn set_h(&mut self, h: T);

    /// Get the current status of the solver (Solving, Complete, Error, etc.).
    fn status(&self) -> &Status<T, V, D>;

    /// Set the status of the solver.
    fn set_status(&mut self, status: Status<T, V, D>);
}
