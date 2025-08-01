//! Solve SDE function

use crate::{
    ControlFlag, Error, Solution, Status,
    interpolate::Interpolation,
    sde::{SDE, StochasticNumericalMethod},
    solout::*,
    traits::{CallBackData, Real, State},
};

/// Solves a Stochastic Differential Equation (SDE) for a system of stochastic differential equations.
///
/// This is the core solution function that drives the numerical integration of SDEs.
/// It handles initialization, time stepping, event detection, and solution output
/// according to the provided output strategy.
///
/// Note that it is recommended to use the `SDEProblem` struct to solve SDEs,
/// as it provides a more feature-rich and convenient interface which
/// wraps this function. See examples on github for more details.
///
/// # Overview
///
/// A Stochastic Differential Equation takes the form:
///
/// ```text
/// dY = a(t, Y)dt + b(t, Y)dW,  t ∈ [t0, tf],  Y(t0) = y0
/// ```
///
/// where:
/// - a(t, Y) is the drift term (deterministic part)
/// - b(t, Y) is the diffusion term (stochastic part)
/// - dW represents a Wiener process increment
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
/// * `solver` - Configured solver instance with appropriate settings (e.g., step size)
/// * `system` - The SDE system that implements the `SDE` trait
/// * `t0` - Initial time point
/// * `tf` - Final time point (can be less than `t0` for backward integration)
/// * `y0` - Initial state vector
/// * `solout` - Solution output strategy that controls which points are included in the result
///
/// # Returns
///
/// * `Ok(Solution)` - If integration completes successfully or is terminated by an event
/// * `Err(Status)` - If an error occurs (e.g., maximum steps reached)
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
///     sde::solve_sde,
///     solout::DefaultSolout,
/// };
/// use nalgebra::SVector;
/// use rand::SeedableRng;
/// use rand_distr::{Distribution, Normal};
///
/// struct GBM {
///     rng: rand::rngs::StdRng,
/// }
///
/// impl GBM {
///     fn new(seed: u64) -> Self {
///         Self {
///             rng: rand::rngs::StdRng::seed_from_u64(seed),
///         }
///     }
/// }
///
/// impl SDE<f64, SVector<f64, 1>> for GBM {
///     fn drift(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
///         dydt[0] = 0.1 * y[0]; // μS
///     }
///     
///     fn diffusion(&self, _t: f64, y: &SVector<f64, 1>, dydw: &mut SVector<f64, 1>) {
///         dydw[0] = 0.2 * y[0]; // σS
///     }
///     
///     fn noise(&mut self, dt: f64, dw: &mut SVector<f64, 1>) {
///         let normal = Normal::new(0.0, dt.sqrt()).unwrap();
///         dw[0] = normal.sample(&mut self.rng);
///     }
/// }
///
/// let t0 = 0.0;
/// let tf = 1.0;
/// let y0 = SVector::<f64, 1>::new(100.0);
/// let mut gbm = GBM::new(42);
/// let mut solver = ExplicitRungeKutta::euler(0.01);
/// let mut solout = DefaultSolout::new();
///
/// // Solve the SDE
/// let result = solve_sde(&mut solver, &mut gbm, t0, tf, &y0, &mut solout);
/// ```
///
/// # Notes
///
/// * For forward integration, `tf` should be greater than `t0`.
/// * For backward integration, `tf` should be less than `t0`.
/// * The `tf == t0` case is considered an error (no integration to perform).
/// * The output points depend on the chosen `Solout` implementation.
/// * Due to the stochastic nature, each run will produce different results unless a specific seed is used.
///
pub fn solve_sde<T, V, D, S, F, O>(
    solver: &mut S,
    sde: &mut F,
    t0: T,
    tf: T,
    y0: &V,
    solout: &mut O,
) -> Result<Solution<T, V, D>, Error<T, V>>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: SDE<T, V, D>,
    S: StochasticNumericalMethod<T, V, D> + Interpolation<T, V>,
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
    match solver.init(sde, t0, tf, y0) {
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
            match solver.init(sde, tm, tf, &ym) {
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

    // For event
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate before starting incase the initial conditions trigger it
    match sde.event(t0, y0) {
        ControlFlag::Continue => {}
        ControlFlag::ModifyState(tm, ym) => {
            // Reinitialize the solver with the modified state
            match solver.init(sde, tm, tf, &ym) {
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

    // Set StochasticNumericalMethod to Solving
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
        match solver.step(sde) {
            Ok(evals) => {
                // Update function evaluations
                solution.evals += evals;
                solution.steps.accepted += 1;
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
                match solver.init(sde, tm, tf, &ym) {
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
        match sde.event(solver.t(), solver.y()) {
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
                    match sde.event(t_guess, &y) {
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
                        match solver.init(sde, tm, tf, &ym) {
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

    // Solution completed successfully
    solver.set_status(Status::Complete);

    // Finalize the solution
    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
