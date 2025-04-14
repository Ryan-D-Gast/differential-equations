//! Numerical Methods for solving ordinary differential equations (ODEs).

use crate::{
    Error, Status,
    ode::ODE,
    traits::{Real, CallBackData},
};
use nalgebra::SMatrix;

/// Type alias for the number of function evaluations
pub type NumEvals = usize;

/// NumericalMethod Trait for ODE NumericalMethods
///
/// ODE NumericalMethods implement this trait to solve ordinary differential equations.
/// This step function is called iteratively to solve the ODE.
/// By implementing this trait, different functions can use a user provided
/// ODE solver to solve the ODE that fits their requirements.
///
pub trait NumericalMethod<T, const R: usize, const C: usize, D = String>
where
    T: Real,
    D: CallBackData,
{
    /// Initialize NumericalMethod before solving ODE
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    ///
    /// # Returns
    /// * Result<(), Status<T, R, C, D>> - Ok if initialization is successful,
    ///
    fn init<F>(
        &mut self,
        ode: &F,
        t0: T,
        tf: T,
        y: &SMatrix<T, R, C>,
    ) -> Result<NumEvals, Error<T, R, C>>
    where
        F: ODE<T, R, C, D>;

    /// Step through solving the ODE by one step
    ///
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    ///
    /// # Returns
    /// * Result<usize, Status<T, R, C, D>> - Ok if step is successful with the number of function evaluations,
    ///
    fn step<F>(&mut self, ode: &F) -> Result<NumEvals, Error<T, R, C>>
    where
        F: ODE<T, R, C, D>;

    // Access fields of the solver

    /// Access time of last accepted step
    fn t(&self) -> T;

    /// Access solution of last accepted step
    fn y(&self) -> &SMatrix<T, R, C>;

    /// Access time of previous accepted step
    fn t_prev(&self) -> T;

    /// Access solution of previous accepted step
    fn y_prev(&self) -> &SMatrix<T, R, C>;

    /// Access step size of next step
    fn h(&self) -> T;

    /// Set step size of next step
    fn set_h(&mut self, h: T);

    /// Status of solver
    fn status(&self) -> &Status<T, R, C, D>;

    /// Set status of solver
    fn set_status(&mut self, status: Status<T, R, C, D>);
}