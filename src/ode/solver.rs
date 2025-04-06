//! Solver Trait for ODE Solvers

use crate::ode::{ODE, EventData};
use crate::interpolate::InterpolationError;
use crate::traits::Real;
use nalgebra::SMatrix;
use std::fmt::{Display, Debug};

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
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &SMatrix<T, R, C>) -> Result<(), SolverError<T, R, C>>
    where
        F: ODE<T, R, C, E>;

    /// Step through solving the ODE by one step
    /// 
    /// # Arguments
    /// * `system` - System of ODEs to solve.
    /// 
    /// # Returns
    /// * Result<usize, SolverStatus<T, R, C, E>> - Ok if step is successful with the number of function evaluations,
    /// 
    fn step<F>(&mut self, ode: &F) -> Result<(), SolverError<T, R, C>>
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

    /// Access the number of function evaluations
    fn evals(&self) -> usize;

    /// Access step size of next step
    fn h(&self) -> T;

    /// Set step size of next step
    fn set_h(&mut self, h: T);

    /// Status of solver
    fn status(&self) -> &SolverStatus<T, R, C, E>;

    /// Set status of solver
    fn set_status(&mut self, status: SolverStatus<T, R, C, E>);
}

/// Solver Error for ODE Solvers
/// 
/// # Variants
/// * `BadInput` - Solver input was bad.
/// * `MaxSteps` - Solver reached maximum steps.
/// * `StepSize` - Solver terminated due to step size converging too small of a value.
/// * `Stiffness` - Solver terminated due to stiffness.
/// 
#[derive(Debug, PartialEq, Clone)]
pub enum SolverError<T, const R: usize, const C: usize>
where
    T: Real,
{
    /// Solver input was bad
    BadInput(String), // During solver.init, if input is bad, return this with reason
    MaxSteps(T, SMatrix<T, R, C>), // If the solver reaches the maximum number of steps
    StepSize(T, SMatrix<T, R, C>), // If the solver step size converges to zero and/or becomes smaller then machine epsilon
    Stiffness(T, SMatrix<T, R, C>), // If the solver detects stiffness e.g. step size converging and/or repeated rejected steps unable to progress
}

impl<T, const R: usize, const C: usize> Display for SolverError<T, R, C>
where
    T: Real + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadInput(msg) => write!(f, "Bad Input: {}", msg),
            Self::MaxSteps(t, y) => write!(f, "Maximum steps reached at (t, y) = ({}, {})", t, y),
            Self::StepSize(t, y) => write!(f, "Step size too small at (t, y) = ({}, {})", t, y),
            Self::Stiffness(t, y) => write!(f, "Stiffness detected at (t, y) = ({}, {})", t, y),
        }
    }
}

/// Solver Status for ODE Solvers
///
/// # Variants
/// * `Uninitialized` - Solver has not been initialized.
/// * `Initialized`   - Solver has been initialized.
/// * `Error`         - Solver encountered an error.
/// * `Solving`       - Solver is solving.
/// * `RejectedStep`  - Solver rejected step.
/// * `Interrupted`    - Solver was interrupted by event with reason.
/// * `Complete`      - Solver completed.
/// 
#[derive(Debug, PartialEq, Clone)]
pub enum SolverStatus<T, const R: usize, const C: usize, E> 
where 
    T: Real,
    E: EventData
{
    Uninitialized,  // Solvers default to this until solver.init is called
    Initialized,    // After solver.init is called
    Error(SolverError<T, R, C>),  // If the solver encounters an error, this status is set so solver status is indicated that an error.
    Solving,        // While the ODE is being solved
    RejectedStep,   // If the solver rejects a step, in this case it will repeat with new smaller step size typically, will return to Solving once the step is accepted
    Interrupted(E), // If the solver is interrupted by event with reason
    Complete,       // If the solver is solving and has reached the final time of the IMatrix<T, R, C, S>P then Complete is returned to indicate such.
}

impl<T, const R: usize, const C: usize, E> Display for SolverStatus<T, R, C, E>
where
    T: Real + Display,
    E: EventData + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "Solver: Uninitialized"),
            Self::Initialized => write!(f, "Solver: Initialized"),
            Self::Error(err) => write!(f, "Solver Error: {}", err),
            Self::Solving => write!(f, "Solver: Solving in progress"),
            Self::RejectedStep => write!(f, "Solver: Step rejected"),
            Self::Interrupted(reason) => write!(f, "Solver: Interrupted - {}", reason),
            Self::Complete => write!(f, "Solver: Complete"),
        }
    }
}