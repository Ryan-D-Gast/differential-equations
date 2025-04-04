//! Solver Trait for ODE Solvers

use crate::ode::{ODE, EventData};
use crate::traits::Real;
use nalgebra::SMatrix;

/// Solver Trait for ODE Solvers
/// 
/// ODE Solvers implement this trait to solve ordinary differential equations.
/// This step function is called iteratively to solve the ODE.
/// By implementing this trait, different functions can use a user provided
/// ODE solver to solve the ODE that fits their requirements.
/// 
pub trait Solver<T, const R: usize, const C: usize, E = String>
where
    T: Real,
    E: EventData,
{
    /// Initialize Solver before solving ODE
    /// 
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// * `t0`     - Initial time.
    /// * `tf`     - Final time.
    /// * `y`      - Initial state.
    /// 
    /// # Returns
    /// * Result<(), SolverStatus<T, R, C, E>> - Ok if initialization is successful,
    /// 
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &SMatrix<T, R, C>) -> Result<(), SolverStatus<T, R, C, E>>
    where
        F: ODE<T, R, C, E>;

    /// Step through solving the ODE by one step
    /// 
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// 
    fn step<F>(&mut self, ode: &F)
    where
        F: ODE<T, R, C, E>;

    /// Interpolate solution between previous and current step
    /// 
    /// # Arguments
    /// * `t_interp` - Time to interpolate to.
    /// 
    /// # Returns
    /// * Interpolated state vector at t_interp.
    /// 
    fn interpolate(&mut self, t_interp: T) -> Result<SMatrix<T, R, C>, InterpolationError<T, R, C>>;

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

    // Statistics of the solver

    /// Number of function evaluations
    fn evals(&self) -> usize;

    /// Number of steps
    fn steps(&self) -> usize;

    /// Number of rejected steps default to 0 so fixed step solvers can ignore this trait
    fn rejected_steps(&self) -> usize { 0 }

    /// Number of accepted steps, default to steps - rejected_steps so fixed step solvers can ignore this trait
    fn accepted_steps(&self) -> usize { self.steps() - self.rejected_steps() }

    /// Status of solver
    fn status(&self) -> &SolverStatus<T, R, C, E>;

    /// Set status of solver
    fn set_status(&mut self, status: SolverStatus<T, R, C, E>);
}

/// Solver Status for ODE Solvers
///
/// # Variants
/// * `Uninitialized` - Solver has not been initialized.
/// * `BadInput`      - Solver input was bad.
/// * `Initialized`   - Solver has been initialized.
/// * `Solving`       - Solver is solving.
/// * `RejectedStep`  - Solver rejected step.
/// * `MaxSteps`      - Solver reached maximum steps.
/// * `StepSize`      - Solver terminated due to step size converging too to small of a value.
/// * `Stiffness`     - Solver terminated due to stiffness.
/// * `Interrupted`    - Solver was interrupted by event with reason.
/// * `Complete`      - Solver completed.
/// 
#[derive(Debug, PartialEq, Clone)]
pub enum SolverStatus<T, const R: usize, const C: usize, E> 
where 
    T: Real,
    E: EventData
{
    Uninitialized, // Solvers default to this until solver.init is called
    BadInput(String), // During solver.init, if input is bad, return this with reason
    Initialized, // After solver.init is called
    Solving, // While the ODE is being solved
    RejectedStep, // If the solver rejects a step, in this case it will repeat with new smaller step size typically, will return to Solving once the step is accepted
    MaxSteps(T, SMatrix<T, R, C>), // If the solver reaches the maximum number of steps
    StepSize(T, SMatrix<T, R, C>), // If the solver step size converges to zero / becomes smaller then T::default_epsilon (machine default_epsilon)
    Stiffness(T, SMatrix<T, R, C>), // If the solver detects stiffness e.g. step size converging and/or repeated rejected steps unable to progress
    Interrupted(E), // If the solver is interrupted by event with reason
    Complete, // If the solver is solving and has reached the final time of the IMatrix<T, R, C, S>P then Complete is returned to indicate such.
}

/// Interpolation Error for ODE Solvers
/// 
/// # Variants
/// * `OutOfBounds` - Given t is not within the previous and current step.
/// 
#[derive(Debug, PartialEq, Clone)]
pub enum InterpolationError<T, const R: usize, const C: usize> 
where 
    T: Real
{
    /// Given t is not within the previous and current step
    OutOfBounds(T, T, T), // t is not within the previous and current step returns the t, t_prev, t_curr
}