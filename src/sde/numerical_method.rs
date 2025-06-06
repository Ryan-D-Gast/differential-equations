//! Numerical Methods for solving stochastic differential equations (SDEs).

use crate::{
    Error, Status,
    sde::SDE,
    alias::Evals,
    traits::{CallBackData, Real, State},
};

/// SDENumericalMethod Trait for SDE NumericalMethods
///
/// SDE NumericalMethods implement this trait to solve stochastic differential equations.
/// This step function is called iteratively to solve the SDE.
/// By implementing this trait, different functions can use a user provided
/// SDE solver to solve the SDE that fits their requirements.
///
pub trait SDENumericalMethod<T, V, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Initialize SDENumericalMethod before solving SDE
    ///
    /// # Arguments
    /// * `sde`    - System of SDEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    ///
    /// # Returns
    /// * Result<NumEvals, Error<T, V>> - Ok with number of evaluations if initialization is successful
    ///
    fn init<F>(&mut self, sde: &mut F, t0: T, tf: T, y: &V) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>;

    /// Step through solving the SDE by one step
    ///
    /// # Arguments
    /// * `sde`    - System of SDEs to solve.
    ///
    /// # Returns
    /// * Result<NumEvals, Error<T, V>> - Ok with number of evaluations if step is successful
    ///
    fn step<F>(&mut self, sde: &mut F) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>;

    // Access fields of the solver

    /// Access time of last accepted step
    fn t(&self) -> T;

    /// Access solution of last accepted step
    fn y(&self) -> &V;

    /// Access time of previous accepted step
    fn t_prev(&self) -> T;

    /// Access solution of previous accepted step
    fn y_prev(&self) -> &V;

    /// Access step size of next step
    fn h(&self) -> T;

    /// Set step size of next step
    fn set_h(&mut self, h: T);

    /// Status of solver
    fn status(&self) -> &Status<T, V, D>;

    /// Set status of solver
    fn set_status(&mut self, status: Status<T, V, D>);
}
