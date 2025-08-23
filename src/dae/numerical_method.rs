//! Numerical Methods for solving differential algebraic equations (DAEs).

use crate::{
    dae::DAE,
    error::Error,
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
};

/// AlgebraicNumericalMethod Trait for DAE NumericalMethods
///
/// DAE NumericalMethods implement this trait to solve differential algebraic equations.
/// This step function is called iteratively to solve the DAE.
/// By implementing this trait, different functions can use a user provided
/// DAE solver to solve the DAE that fits their requirements.
///
pub trait AlgebraicNumericalMethod<T, V, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Initialize AlgebraicNumericalMethod before solving DAE
    ///
    /// # Arguments
    /// * `system` - System of DAEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, V>> - Ok if initialization is successful,
    ///
    fn init<F>(&mut self, dae: &F, t0: T, tf: T, y: &V) -> Result<Evals, Error<T, V>>
    where
        F: DAE<T, V, D>;

    /// Step through solving the DAE by one step
    ///
    /// # Arguments
    /// * `system` - System of DAEs to solve.
    ///
    /// # Returns
    /// * Result<Evals, Errors<T, V>> - Ok if step is successful with the number of function evaluations,
    ///
    fn step<F>(&mut self, dae: &F) -> Result<Evals, Error<T, V>>
    where
        F: DAE<T, V, D>;

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
