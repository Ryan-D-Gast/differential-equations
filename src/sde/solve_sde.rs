//! Solve SDE function

use crate::{
    ControlFlag, Error, Solution, Status,
    interpolate::Interpolation,
    sde::{SDE, NumericalMethod},
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
///     fn noise(&self, dt: f64, dw: &mut SVector<f64, 1>) {
///         let normal = Normal::new(0.0, dt.sqrt()).unwrap();
///         dw[0] = normal.sample(&mut self.rng.clone());
///     }
/// }
///
/// let t0 = 0.0;
/// let tf = 1.0;
/// let y0 = SVector::<f64, 1>::new(100.0);
/// let gbm = GBM::new(42);
/// let mut solver = EM::new(0.01);
/// let mut solout = DefaultSolout::new();
///
/// // Solve the SDE
/// let result = solve_sde(&mut solver, &gbm, t0, tf, &y0, &mut solout);
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
    sde: &F,
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
    S: NumericalMethod<T, V, D> + Interpolation<T, V>,
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
        ControlFlag::Terminate(reason) => {
            solution.status = Status::Interrupted(reason);
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // Set NumericalMethod to Solving
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
        match solver.step(sde) {
            Ok(evals) => {
                // Update function evaluations
                solution.evals += evals;
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
            ControlFlag::Terminate(re) => {
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
                        ControlFlag::Terminate(re) => {
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
                solution.status = Status::Interrupted(reason);
                solution.timer.complete();

                return Ok(solution);
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
