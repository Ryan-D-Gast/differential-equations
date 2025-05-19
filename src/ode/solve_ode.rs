//! Solve ODEProblem function

use crate::{
    ControlFlag, Error, Solution, Status,
    interpolate::Interpolation,
    ode::{ODE, numerical_method::ODENumericalMethod},
    solout::*,
    traits::{CallBackData, Real, State},
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
/// let mut method = DOP853::new().rtol(1e-8).atol(1e-10);
/// let mut solout = DefaultSolout::new();
/// let system = ExponentialGrowth;
/// let y0 = 1.0;
/// let result = solve_ode(&mut method, &system, 0.0, 1.0, &y0, &mut solout);
///
/// match result {
///     Ok(solution) => {
///         println!("Final value: {}", solution.y.last().unwrap());
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
pub fn solve_ode<T, V, D, S, F, O>(
    solver: &mut S,
    ode: &F,
    t0: T,
    tf: T,
    y0: &V,
    solout: &mut O,
) -> Result<Solution<T, V, D>, Error<T, V>>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    S: ODENumericalMethod<T, V, D> + Interpolation<T, V>,
    O: Solout<T, V, D>,
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
            solution.evals += evals.fcn;
            solution.jac_evals += evals.jac;
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
                    solution.evals += evals.fcn;
                    solution.jac_evals += evals.jac;
                }
                Err(e) => return Err(e),
            }
        }
        ControlFlag::Terminate(reason) => {
            solution.status = Status::Interrupted(reason);
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // For event
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate before starting incase the initial conditions trigger it
    match ode.event(t0, y0) {
        ControlFlag::Continue => {}
        ControlFlag::ModifyState(tm, ym) => {
            // Reinitialize the solver with the modified state
            match solver.init(ode, tm, tf, &ym) {
                Ok(evals) => {
                    solution.evals += evals.fcn;
                    solution.jac_evals += evals.jac;
                }
                Err(e) => return Err(e),
            }
        }
        ControlFlag::Terminate(reason) => {
            solution.status = Status::Interrupted(reason);
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // Set ODENumericalMethod to Solving
    solver.set_status(Status::Solving);
    solution.status = Status::Solving;

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
        solution.steps += 1;
        match solver.step(ode) {
            Ok(evals) => {
                // Update function evaluations
                solution.evals += evals.fcn;
                solution.jac_evals += evals.jac;

                // Check for a RejectedStep
                if let Status::RejectedStep = solver.status() {
                    // Update rejected steps and re-do the step
                    solution.rejected_steps += 1;
                    continue;
                } else {
                    // Update accepted steps and continue to processing
                    solution.accepted_steps += 1;
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
                        solution.evals += evals.fcn;
                        solution.jac_evals += evals.jac;
                    }
                    Err(e) => return Err(e),
                }
            }
            ControlFlag::Terminate(reason) => {
                solution.status = Status::Interrupted(reason);
                solution.timer.complete();
                return Ok(solution);
            }
        }

        // Check event condition
        match ode.event(solver.t(), solver.y()) {
            ControlFlag::Continue => {
                // Update last continue point
                tc = solver.t();
            }
            // Any non-continue flag means we need to root-find for the precise event time
            evt @ (ControlFlag::ModifyState(_, _) | ControlFlag::Terminate(_)) => {
                // Store the initial event that was detected
                let initial_event = evt;
                
                // Update last event point
                ts = solver.t();

                // Root-finding to determine the precise point where event is triggered
                // Method: Regula Falsi (False Position) with Illinois adjustment
                let mut side_count = 0; // Illinois method counter

                // For Illinois method adjustment
                let mut f_low: T = T::from_f64(-1.0).unwrap(); // Continue represented as -1
                let mut f_high: T = T::from_f64(1.0).unwrap(); // Event represented as +1
                let mut t_guess: T;
                
                // The final event we'll detect (might change during root finding)
                let mut final_event = initial_event.clone();

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
                        ControlFlag::Continue => {
                            tc = t_guess;

                            // Illinois adjustment to improve convergence
                            side_count += 1;
                            if side_count >= 2 {
                                f_high /= T::from_f64(2.0).unwrap(); // Reduce influence of high point
                                side_count = 0;
                            }
                        }
                        // Any non-continue flag indicates we've found an event
                        evt @ (ControlFlag::ModifyState(_, _) | ControlFlag::Terminate(_)) => {
                            final_event = evt; // Update the event we found
                            ts = t_guess;      // Update the event time
                            side_count = 0;
                            f_low = T::from_f64(-1.0).unwrap(); // Reset low point influence
                        }
                    }
                }

                // Get the final event point
                let event_time = ts;
                let event_state = solver.interpolate(ts).unwrap();
                
                // Remove points after the event point incase solout wrote them
                // Find the cutoff index based on integration direction
                let cutoff_index = if integration_direction > T::zero() {
                    // Forward integration - find first index where t > event_time
                    solution.t.iter().position(|&t| t > event_time)
                } else {
                    // Backward integration - find first index where t < event_time
                    solution.t.iter().position(|&t| t < event_time)
                };
                
                // If we found a cutoff point, truncate both vectors
                if let Some(idx) = cutoff_index {
                    solution.truncate(idx);
                }

                // Now handle the event based on its type
                match final_event {
                    ControlFlag::ModifyState(tm, ym) => {
                        // Record the modified state point
                        solution.push(tm, ym.clone());
                        
                        // Reinitialize the solver with the modified state at the precise time
                        match solver.init(ode, tm, tf, &ym) {
                            Ok(evals) => {
                                solution.evals += evals.fcn;
                                solution.jac_evals += evals.jac;
                            }
                            Err(e) => return Err(e),
                        }
                        
                        // Update tc to the event time
                        tc = event_time;
                    }
                    ControlFlag::Terminate(reason) => {
                        // Add the event point
                        solution.push(event_time, event_state);
                        
                        // Set solution parameters
                        solution.status = Status::Interrupted(reason);
                        solution.timer.complete();
                        
                        return Ok(solution);
                    }
                    ControlFlag::Continue => {
                        // This shouldn't happen, but if it does, just continue
                        tc = event_time;
                    }
                }
            }
        }
    }

    // Solution completed successfully
    solver.set_status(Status::Complete);

    // Finalize the solution
    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
