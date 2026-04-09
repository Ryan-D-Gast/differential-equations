//! Solve SDE function

use crate::{
    control::ControlFlag,
    error::Error,
    interpolate::Interpolation,
    sde::{SDE, StochasticNumericalMethod},
    solout::*,
    solution::Solution,
    status::Status,
    traits::{Real, State},
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
/// * `t`      - Vector of time points
/// * `y`      - Vector of state vectors at each time point
/// * `solout` - Struct of the solution output strategy used
/// * `status` - Final solver status (Complete or Interrupted)
/// * `evals`  - Number of function evaluations performed
/// * `steps`  - Total number of steps attempted
/// * `timer`  - Timer object for tracking solve time
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
pub fn solve_sde<T, Y, S, F, O>(
    solver: &mut S,
    sde: &mut F,
    t0: T,
    tf: T,
    y0: &Y,
    solout: &mut O,
) -> Result<Solution<T, Y>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    F: SDE<T, Y>,
    S: StochasticNumericalMethod<T, Y> + Interpolation<T, Y>,
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
        ControlFlag::Terminate => {
            solution.status = Status::Interrupted;
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // Set StochasticNumericalMethod to Solving
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
