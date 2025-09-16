//! Solve ODEProblem function

use crate::{
    control::ControlFlag,
    error::Error,
    interpolate::Interpolation,
    ode::{ODE, OrdinaryNumericalMethod},
    solout::*,
    solution::Solution,
    status::Status,
    traits::{Real, State},
};

/// Solves an Initial Value Problem (ODEProblem) for a system of ordinary differential equations.
///
/// This is the core solution function that drives the numerical integration of ODEs.
/// It handles initialization, time stepping, event detection, and solution output
/// according to the provided output strategy.
///
/// Note that it is recommend to use the `ODEProblem` struct to solve the ODEs,
/// as it provides far more feature rich and convenient interface which
/// wraps this function. See examples on github for more details.
///
/// # Overview
///
/// An Initial Value Problem takes the form:
///
/// ```text
/// dy/dt = f(t, y),  t ∈ [t0, tf],  y(t0) = y0
/// ```
///
/// This function solves such a problem by:
///
/// 1. Initializing the solver with the system and initial conditions
/// 2. Stepping the solver through the integration interval
/// 3. Detecting and handling events (if any)
/// 4. Collecting solution points according to the specified output strategy
/// 5. Monitoring for errors or exceptional conditions
///
/// # Arguments
///
/// * `solver` - Configured solver instance with appropriate settings (e.g., tolerances)
/// * `system` - The ODE system that implements the `ODE` trait
/// * `t0` - Initial time point
/// * `tf` - Final time point (can be less than `t0` for backward integration)
/// * `y0` - Initial state vector
/// * `solout` - Solution output strategy that controls which points are included in the result
///
/// # Returns
///
/// * `Ok(Solution)` - If integration completes successfully or is terminated by an event
/// * `Err(Status)` - If an error occurs (e.g., excessive stiffness, maximum steps reached)
///
/// # Solution Object
///
/// The returned `Solution` object contains:
///
/// * `t`      - Vector of time points
/// * `y`      - Vector of state vectors at each time point
/// * `solout` - Struct of the solution output strategy used
/// * `status` - Final solver status (Complete or Interrupted)
/// * `evals`  - Number of function evaluations performed
/// * `steps`  - Total number of steps attempted
/// * `timer`  - Timer for tracking the solve time
///
/// # Event Handling
///
/// The solver checks for events after each step using the `event` method of the system.
/// If an event returns `ControlFlag::Terminate`, the integration stops and interpolates
/// to find the precise point where the event occurred, using a modified regula falsi method.
///
/// # Examples
///
/// ```
/// use differential_equations::{
///     prelude::*,
///     solout::DefaultSolout,
///     ode::solve_ode,
/// };
/// use nalgebra::Vector1;
///
/// // Define a simple exponential growth ode: dy/dt = y
/// struct ExponentialGrowth;
///
/// impl ODE for ExponentialGrowth {
///     fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
///         *dydt = *y;
///     }
/// }
///
/// // Solve from t=0 to t=1 with initial condition y=1
/// let mut method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
/// let mut solout = DefaultSolout::new();
/// let system = ExponentialGrowth;
/// let y0 = 1.0;
/// let result = solve_ode(&mut method, &system, 0.0, 1.0, &y0, &mut solout);
///
/// match result {
///     Ok(solution) => {
///         println!("Final value: {}", solution.y.last().unwrap());
///         println!("Number of steps: {}", solution.steps.total());
///     },
///     Err(status) => {
///         println!("Integration failed: {:?}", status);
///     }
/// }
/// ```
///
/// # Notes
///
/// * For forward integration, `tf` should be greater than `t0`.
/// * For backward integration, `tf` should be less than `t0`.
/// * The `tf == t0` case is considered an error (no integration to perform).
/// * The output points depend on the chosen `Solout` implementation.
///
pub fn solve_ode<T, Y, S, F, O>(
    solver: &mut S,
    ode: &F,
    t0: T,
    tf: T,
    y0: &Y,
    solout: &mut O,
) -> Result<Solution<T, Y>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
    S: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    O: Solout<T, Y>,
{
    // Initialize the Solution object
    let mut solution = Solution::new();

    // Begin timing the solution process
    solution.timer.start();

    // Determine integration direction and check that tf != t0
    let integration_direction = match (tf - t0).signum() {
        x if x == T::one() => T::one(),
        x if x == T::from_f64(-1.0).unwrap() => T::from_f64(-1.0).unwrap(),
        _ => {
            return Err(Error::BadInput {
                msg: "Final time tf must be different from initial time t0.".to_string(),
            });
        }
    };

    // Clear statistics in case it was used before and reset solver and check for errors
    match solver.init(ode, t0, tf, y0) {
        Ok(evals) => {
            solution.evals += evals;
        }
        Err(e) => return Err(e),
    }

    // Call solout to initialize the output strategy
    let mut y_curr = *solver.y();
    let mut y_prev = *solver.y_prev();
    match solout.solout(
        solver.t(),
        solver.t_prev(),
        &y_curr,
        &y_prev,
        solver,
        &mut solution,
    ) {
        ControlFlag::Continue => {}
        ControlFlag::ModifyState(tm, ym) => {
            // Reinitialize the solver with the modified state
            match solver.init(ode, tm, tf, &ym) {
                Ok(evals) => {
                    solution.evals += evals;
                }
                Err(e) => return Err(e),
            }
        }
        ControlFlag::Terminate => {
            solution.status = Status::Interrupted;
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // Set OrdinaryNumericalMethod to Solving
    solver.set_status(Status::Solving);
    solution.status = Status::Solving;

    // Main Loop
    loop {
        // Check if next step overshoots tf
        if (solver.t() + solver.h() - tf) * integration_direction > T::zero() {
            // New step size to reach tf
            let h_new = tf - solver.t();

            // If the new step size is extremely small, consider the integration complete
            if h_new.abs() < T::default_epsilon() * T::from_f64(10.0).unwrap() {
                // Set the status to complete and finalize the solution
                solver.set_status(Status::Complete);
                solution.status = Status::Complete;
                solution.timer.complete();
                return Ok(solution);
            }

            // Correct step size to reach tf
            solver.set_h(h_new);
        }

        // Perform a step
        match solver.step(ode) {
            Ok(evals) => {
                // Update function evaluations
                solution.evals += evals;

                // Check for a RejectedStep
                if let Status::RejectedStep = solver.status() {
                    // Update rejected steps and re-do the step
                    solution.steps.rejected += 1;
                    continue;
                } else {
                    // Update accepted steps and continue to processing
                    solution.steps.accepted += 1;
                }
            }
            Err(e) => {
                // Set solver status to error and return error
                return Err(e);
            }
        }

        // Record the result
        y_curr = *solver.y();
        y_prev = *solver.y_prev();
        match solout.solout(
            solver.t(),
            solver.t_prev(),
            &y_curr,
            &y_prev,
            solver,
            &mut solution,
        ) {
            ControlFlag::Continue => {}
            ControlFlag::ModifyState(tm, ym) => {
                // Reinitialize the solver with the modified state
                match solver.init(ode, tm, tf, &ym) {
                    Ok(evals) => {
                        solution.evals += evals;
                    }
                    Err(e) => return Err(e),
                }
            }
            ControlFlag::Terminate => {
                solution.status = Status::Interrupted;
                solution.timer.complete();
                return Ok(solution);
            }
        }

        // If we've essentially reached tf, exit the loop and finalize
        if (tf - solver.t()).abs() <= T::default_epsilon() * T::from_f64(10.0).unwrap() {
            break;
        }
    }

    // Solution completed successfully
    solver.set_status(Status::Complete);

    // Finalize the solution
    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
