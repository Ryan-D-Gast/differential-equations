//! Solve IVP function

use crate::ode::{
    EventAction, EventData, ODE, Solout, Solution, Solver, SolverError, SolverStatus,
};
use crate::traits::Real;
use nalgebra::SMatrix;
use std::time::Instant;

/// Solves an Initial Value Problem (IVP) for a system of ordinary differential equations.
///
/// This is the core solution function that drives the numerical integration of ODEs.
/// It handles initialization, time stepping, event detection, and solution output
/// according to the provided output strategy.
///
/// Note that it is recommend to use the `IVP` struct to solve the ODEs,
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
/// * `Err(SolverStatus)` - If an error occurs (e.g., excessive stiffness, maximum steps reached)
///
/// # Solution Object
///
/// The returned `Solution` object contains:
///
/// * `t` - Vector of time points
/// * `y` - Vector of state vectors at each time point
/// * `solout` - Struct of the solution output strategy used
/// * `status` - Final solver status (Complete or Interrupted)
/// * `evals` - Number of function evaluations performed
/// * `steps` - Total number of steps attempted
/// * `rejected_steps` - Number of steps rejected by the error control
/// * `accepted_steps` - Number of steps accepted by the error control
/// * `solve_time` - Wall time taken for the integration
///
/// # Event Handling
///
/// The solver checks for events after each step using the `event` method of the system.
/// If an event returns `EventAction::Terminate`, the integration stops and interpolates
/// to find the precise point where the event occurred, using a modified regula falsi method.
///
/// # Examples
///
/// ```
/// use differential_equations::ode::*;
/// use differential_equations::ode::solout::DefaultSolout;
/// use nalgebra::Vector1;
///
/// // Define a simple exponential growth ode: dy/dt = y
/// struct ExponentialGrowth;
///
/// impl ODE<f64, 1, 1> for ExponentialGrowth {
///     fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
///         dydt[0] = y[0];
///     }
/// }
///
/// // Solve from t=0 to t=1 with initial condition y=1
/// let mut solver = DOP853::new().rtol(1e-8).atol(1e-10);
/// let mut solout = DefaultSolout::new();
/// let system = ExponentialGrowth;
/// let y0 = Vector1::new(1.0);
/// let result = solve_ivp(&mut solver, &system, 0.0, 1.0, &y0, &mut solout);
///
/// match result {
///     Ok(solution) => {
///         println!("Final value: {}", solution.y.last().unwrap()[0]);
///         println!("Number of steps: {}", solution.steps);
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
pub fn solve_ivp<T, const R: usize, const C: usize, E, S, F, O>(
    solver: &mut S,
    ode: &F,
    t0: T,
    tf: T,
    y0: &SMatrix<T, R, C>,
    solout: &mut O,
) -> Result<Solution<T, R, C, E>, SolverError<T, R, C>>
where
    T: Real,
    E: EventData,
    F: ODE<T, R, C, E>,
    S: Solver<T, R, C, E>,
    O: Solout<T, R, C, E>,
{
    // Timer for measuring solve time
    let start = Instant::now();

    // Initialize the Solution object
    let mut solution = Solution::new();

    // Add initial point to output if include_t0_tf is true
    if solout.include_t0_tf() {
        solution.push(t0, *y0);
    }

    // Determine integration direction and check that tf != t0
    let integration_direction = match (tf - t0).signum() {
        x if x == T::one() => T::one(),
        x if x == T::from_f64(-1.0).unwrap() => T::from_f64(-1.0).unwrap(),
        _ => {
            return Err(SolverError::BadInput(
                "Final time tf must be different from initial time t0.".to_string(),
            ));
        }
    };

    // Clear statistics in case it was used before and reset solver and check for errors
    match solver.init(ode, t0, tf, y0) {
        Ok(evals) => {
            solution.evals += evals;
        }
        Err(e) => return Err(e),
    }

    // For event
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate before starting incase the initial conditions trigger it
    match ode.event(t0, y0) {
        EventAction::Continue => {}
        EventAction::Terminate(reason) => {
            solution.status = SolverStatus::Interrupted(reason.clone());
            solution.solve_time = T::from_f64(start.elapsed().as_secs_f64()).unwrap();
            return Ok(solution);
        }
    }

    // Set Solver to Solving
    solver.set_status(SolverStatus::Solving);

    // Main Loop
    let mut solving = true;
    while solving {
        // Check if next step overshoots tf
        if (solver.t() + solver.h() - tf) * integration_direction > T::zero() {
            // Correct step size to reach tf
            solver.set_h(tf - solver.t());
            solving = false;
        }

        // Perform a step
        match solver.step(ode) {
            Ok(evals) => {
                // Update function evaluations
                solution.evals += evals;
            }
            Err(e) => {
                // Set solver status to error and return error
                return Err(e);
            }
        }
        solution.steps += 1;

        // Check for rejected step
        match solver.status() {
            SolverStatus::Solving => {
                solution.accepted_steps += 1;
            }
            SolverStatus::RejectedStep => {
                solution.rejected_steps += 1;
                continue;
            }
            _ => break,
        }

        // Record the result
        solout.solout(solver, &mut solution);

        // Check event condition
        match ode.event(solver.t(), solver.y()) {
            EventAction::Continue => {
                // Update last continue point
                tc = solver.t();
            }
            EventAction::Terminate(re) => {
                // For iteration to event point
                let mut reason = re;

                // Update last stop point
                ts = solver.t();

                // If event_tolerance is set, interpolate to the point where event is triggered
                // Method: Regula Falsi (False Position) with Illinois adjustment
                let mut side_count = 0; // Illinois method counter

                // For Illinois method adjustment
                let mut f_low: T = T::from_f64(-1.0).unwrap(); // Continue represented as -1
                let mut f_high: T = T::from_f64(1.0).unwrap(); // Terminate represented as +1
                let mut t_guess: T;

                let max_iterations = 20; // Prevent infinite loops
                let tol = T::from_f64(1e-10).unwrap(); // Tolerance for convergence

                // False position method with Illinois adjustment
                for _ in 0..max_iterations {
                    // Check if we've reached desired precision
                    if (ts - tc).abs() <= tol {
                        break;
                    }

                    // False position formula with Illinois adjustment
                    t_guess = (tc * f_high - ts * f_low) / (f_high - f_low);

                    // Protect against numerical issues
                    if !t_guess.is_finite()
                        || (integration_direction > T::zero() && (t_guess <= tc || t_guess >= ts))
                        || (integration_direction < T::zero() && (t_guess >= tc || t_guess <= ts))
                    {
                        t_guess = (tc + ts) / T::from_f64(2.0).unwrap(); // Fall back to bisection
                    }

                    // Interpolate state at guess point
                    let y = solver.interpolate(t_guess).unwrap();

                    // Check event at guess point
                    match ode.event(t_guess, &y) {
                        EventAction::Continue => {
                            tc = t_guess;

                            // Illinois adjustment to improve convergence
                            side_count += 1;
                            if side_count >= 2 {
                                f_high /= T::from_f64(2.0).unwrap(); // Reduce influence of high point
                                side_count = 0;
                            }
                        }
                        EventAction::Terminate(re) => {
                            reason = re;
                            ts = t_guess;
                            side_count = 0;
                            f_low = T::from_f64(-1.0).unwrap(); // Reset low point influence
                        }
                    }
                }

                // Final event point
                let y_final = solver.interpolate(ts).unwrap();

                // Remove points after the event point and add the event point
                // Find the cutoff index based on integration direction
                let cutoff_index = if integration_direction > T::zero() {
                    // Forward integration - find first index where t > ts
                    solution.t.iter().position(|&t| t > ts)
                } else {
                    // Backward integration - find first index where t < ts
                    solution.t.iter().position(|&t| t < ts)
                };

                // If we found a cutoff point, truncate both vectors
                if let Some(idx) = cutoff_index {
                    solution.truncate(idx);
                }

                // Add the event point
                solution.push(ts, y_final);

                // Set solution parameters
                solution.status = SolverStatus::Interrupted(reason.clone());
                solution.solve_time = T::from_f64(start.elapsed().as_secs_f64()).unwrap();

                return Ok(solution);
            }
        }
    }

    // Check for problems in Solver Status
    match solver.status() {
        SolverStatus::Solving => {
            solver.set_status(SolverStatus::Complete);

            // Add final point to output if include_t0_tf is true
            if solout.include_t0_tf() && solution.t.last().copied() != Some(tf) {
                solution.push(tf, *solver.y());
            }

            // Set solution parameters
            solution.status = SolverStatus::Complete;
            solution.solve_time = T::from_f64(start.elapsed().as_secs_f64()).unwrap();

            Ok(solution)
        }
        // Everything below should be unreachable.
        _ => {
            unreachable!()
        }
    }
}
