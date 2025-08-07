//! Numerical Methods for solving ordinary differential equations (ODEs).

use crate::{
    Error, Status,
    ode::ODE,
    stats::Evals,
    traits::{CallBackData, Real, State},
};

/// OrdinaryNumericalMethod Trait for ODE NumericalMethods
///
/// ODE NumericalMethods implement this trait to solve ordinary differential equations.
/// This step function is called iteratively to solve the ODE.
/// By implementing this trait, different functions can use a user provided
/// ODE solver to solve the ODE that fits their requirements.
///
pub trait OrdinaryNumericalMethod<T, Y, D = String>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    /// Initialize OrdinaryNumericalMethod before solving ODE
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    ///
    /// # Returns
    /// * Result<Evals, Error<T, Y>> - Ok if initialization is successful,
    ///
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>;

    /// Step through solving the ODE by one step
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    ///
    /// # Returns
    /// * Result<Evals, Errors<T, Y>> - Ok if step is successful with the number of function evaluations,
    ///
    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>;

    // Access fields of the solver

    /// Access time of last accepted step
    fn t(&self) -> T;

    /// Access solution of last accepted step
    fn y(&self) -> &Y;

    /// Access time of previous accepted step
    fn t_prev(&self) -> T;

    /// Access solution of previous accepted step
    fn y_prev(&self) -> &Y;

    /// Access step size of next step
    fn h(&self) -> T;

    /// Set step size of next step
    fn set_h(&mut self, h: T);

    /// Status of solver
    fn status(&self) -> &Status<T, Y, D>;

    /// Set status of solver
    fn set_status(&mut self, status: Status<T, Y, D>);
}
