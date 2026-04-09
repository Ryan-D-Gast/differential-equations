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
    traits::{Real, State},
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
/// * `Ok(Solution<T, Y>)`: If the integration completes successfully (reaches `tf`) or is
///   cleanly interrupted by an event detected by `dde.event()` or `solout.solout()`.
///   The [`Solution`] struct contains the time points, corresponding state vectors,
///   solver statistics, and potentially event data (`D`).
/// * `Err(Error<T, Y>)`: If the solver encounters an error during initialization or stepping.
///   This could be due to invalid input (`Error::BadInput`), failure to meet tolerances
///   (`Error::StepSizeTooSmall`), exceeding the maximum number of steps (`Error::MaxSteps`),
///   or potential stiffness detected by the solver (`Error::Stiffness`).
///
pub fn solve_dde<const L: usize, T, Y, S, F, H, O>(
    solver: &mut S,
    dde: &F,
    t0: T,
    tf: T,
    y0: &Y,
    phi: H,
    solout: &mut O,
) -> Result<Solution<T, Y>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    F: DDE<L, T, Y>,
    H: Fn(T) -> Y + Clone,
    S: DelayNumericalMethod<L, T, Y, H> + Interpolation<T, Y>,
    O: Solout<T, Y>,
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
        ControlFlag::Terminate => {
            solution.status = Status::Interrupted;
            solution.timer.complete();
            return Ok(solution);
        }
    }

    // Set solver status
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
        match solver.step(dde, &phi) {
            Ok(evals) => {
                solution.evals += evals;

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

    solver.set_status(Status::Complete);

    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
