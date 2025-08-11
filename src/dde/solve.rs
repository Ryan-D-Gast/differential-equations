//! Core solver function for Delay Differential Equation Initial Value Problems.

use crate::{
    control::ControlFlag,
    dde::DDE,
    dde::numerical_method::DelayNumericalMethod,
    error::Error,
    interpolate::Interpolation,
    solout::*,
    solution::Solution,
    status::Status,
    traits::{CallBackData, Real, State},
};

/// Solves an Initial Value Problem for a system of Delay Differential Equations (DDEs).
///
/// This is the core function that drives the numerical integration process for DDEs.
/// It manages the solver steps, handles history lookups, processes events defined
/// in the [`DDE`] trait, and collects the solution according to the [`Solout`] strategy.
///
/// **Note:** Users typically interact with the higher-level [`DDEProblem`] struct,
/// which provides a more convenient interface and wraps this function. Use this
/// function directly only if you need finer control over the solving process
/// beyond what `DDEProblem` offers.
///
/// # Overview
///
/// A DDE Initial Value Problem is defined as:
///
/// ```text
/// dy/dt = f(t, y(t), y(t - tau1), y(t - tau2), ...),  for t in [t0, tf]
/// y(t) = phi(t),                                      for t <= t0
/// ```
///
/// This function solves such problems by iteratively stepping the `solver` from `t0` to `tf`.
///
/// # Arguments
///
/// * `solver`: A mutable reference to a DDE solver instance. The solver must implement
///   both [`DelayNumericalMethod`] (specifically adapted for DDEs, handling the history lookup
///   closure in its `init` and `step` methods) and [`Interpolation`] (for dense output
///   and event localization).
/// * `dde`: A reference to the DDE system definition, which must implement the [`DDE`] trait.
///   This provides the `diff` function (the right-hand side of the DDE) and optionally
///   an `event` function.
/// * `t0`: The initial time point of the integration interval.
/// * `tf`: The final time point of the integration interval. `tf` must be different from `t0`.
/// * `y0`: A reference to the initial state vector `y(t0)`.
/// * `phi`: The initial history function. This must be a function or closure implementing
///   `Fn(T) -> V` that returns the state vector `Y` for any time `t <= t0`.
/// * `solout`: A mutable reference to a solution output handler implementing the [`Solout`] trait.
///   This determines how and when solution points `(t, y)` are recorded or processed.
///
/// # Returns
///
/// * `Ok(Solution<T, Y, D>)`: If the integration completes successfully (reaches `tf`) or is
///   cleanly interrupted by an event detected by `dde.event()` or `solout.solout()`.
///   The [`Solution`] struct contains the time points, corresponding state vectors,
///   solver statistics, and potentially event data (`D`).
/// * `Err(Error<T, Y>)`: If the solver encounters an error during initialization or stepping.
///   This could be due to invalid input (`Error::BadInput`), failure to meet tolerances
///   (`Error::StepSizeTooSmall`), exceeding the maximum number of steps (`Error::MaxSteps`),
///   or potential stiffness detected by the solver (`Error::Stiffness`).
///
pub fn solve_dde<const L: usize, T, Y, D, S, F, H, O>(
    solver: &mut S,
    dde: &F,
    t0: T,
    tf: T,
    y0: &Y,
    phi: H,
    solout: &mut O,
) -> Result<Solution<T, Y, D>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y + Clone,
    S: DelayNumericalMethod<L, T, Y, H, D> + Interpolation<T, Y>,
    O: Solout<T, Y, D>,
{
    // Initialize the Solution object
    let mut solution = Solution::new();
    solution.timer.start();

    // Determine integration direction and check tf != t0
    let integration_direction = match (tf - t0).signum() {
        x if x == T::one() => T::one(),
        x if x == T::from_f64(-1.0).unwrap() => T::from_f64(-1.0).unwrap(),
        _ => {
            return Err(Error::BadInput {
                msg: "Final time tf must be different from initial time t0.".to_string(),
            });
        }
    };

    // Initialize the solver
    match solver.init(dde, t0, tf, y0, &phi) {
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
            match solver.init(dde, tm, tf, &ym, &phi) {
                Ok(evals) => {
                    solution.evals += evals;
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

    // For event detection
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate event at the start
    match dde.event(t0, y0) {
        ControlFlag::Continue => {}
        ControlFlag::ModifyState(tm, ym) => {
            // Reinitialize the solver with the modified state
            match solver.init(dde, tm, tf, &ym, &phi) {
                Ok(evals) => {
                    solution.evals += evals;
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

    // Set solver status
    solver.set_status(Status::Solving);
    solution.status = Status::Solving;

    // Main Loop
    let mut solving = true;
    while solving {
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
            solving = false;
        }

        // Perform a step
        match solver.step(dde, &phi) {
            Ok(evals) => {
                solution.evals += evals;
                solution.steps.accepted += 1;

                if let Status::RejectedStep = solver.status() {
                    solution.steps.rejected += 1;
                    continue;
                } else {
                    solution.steps.accepted += 1;
                }
            }
            Err(e) => {
                return Err(e);
            }
        }

        // Record the result using solout
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
                match solver.init(dde, tm, tf, &ym, &phi) {
                    Ok(evals) => {
                        solution.evals += evals;
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
        match dde.event(solver.t(), solver.y()) {
            ControlFlag::Continue => {
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

                // Track the initial interval size for relative comparison
                let initial_interval = (ts - tc).abs();

                // False position method with Illinois adjustment
                for _ in 0..max_iterations {
                    // Check if we've reached desired precision (relative to initial interval)
                    let current_interval = (ts - tc).abs();
                    if current_interval <= tol * initial_interval {
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
                    match dde.event(t_guess, &y) {
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
                            ts = t_guess; // Update the event time
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

                // Check if event time is very close to the last recorded point and remove it to avoid duplicates
                if !solution.t.is_empty() {
                    let last_t = *solution.t.last().unwrap();
                    let time_diff = (event_time - last_t).abs();

                    // Check if the time difference is within a small tolerance
                    if time_diff <= tol * initial_interval {
                        // Remove the last point (t, y) to avoid very close duplicates
                        solution.pop();
                    }
                }

                // Now handle the event based on its type
                match final_event {
                    ControlFlag::ModifyState(tm, ym) => {
                        // Record the modified state point
                        solution.push(tm, ym);

                        // Reinitialize the solver with the modified state at the precise time
                        match solver.init(dde, tm, tf, &ym, &phi) {
                            Ok(evals) => {
                                solution.evals += evals;
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

    solver.set_status(Status::Complete);

    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
