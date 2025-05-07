//! Core solver function for Delay Differential Equation Initial Value Problems.

use crate::{
    ControlFlag, Error, Solution, Status,
    dde::DDE,
    dde::numerical_method::NumericalMethod,
    interpolate::Interpolation,
    solout::*,
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
///   both [`NumericalMethod`] (specifically adapted for DDEs, handling the history lookup
///   closure in its `init` and `step` methods) and [`Interpolation`] (for dense output
///   and event localization).
/// * `dde`: A reference to the DDE system definition, which must implement the [`DDE`] trait.
///   This provides the `diff` function (the right-hand side of the DDE) and optionally
///   an `event` function.
/// * `t0`: The initial time point of the integration interval.
/// * `tf`: The final time point of the integration interval. `tf` must be different from `t0`.
/// * `y0`: A reference to the initial state vector `y(t0)`.
/// * `phi`: The initial history function. This must be a function or closure implementing
///   `Fn(T) -> V` that returns the state vector `V` for any time `t <= t0`.
/// * `solout`: A mutable reference to a solution output handler implementing the [`Solout`] trait.
///   This determines how and when solution points `(t, y)` are recorded or processed.
///
/// # Returns
///
/// * `Ok(Solution<T, V, D>)`: If the integration completes successfully (reaches `tf`) or is
///   cleanly interrupted by an event detected by `dde.event()` or `solout.solout()`.
///   The [`Solution`] struct contains the time points, corresponding state vectors,
///   solver statistics, and potentially event data (`D`).
/// * `Err(Error<T, V>)`: If the solver encounters an error during initialization or stepping.
///   This could be due to invalid input (`Error::BadInput`), failure to meet tolerances
///   (`Error::StepSizeTooSmall`), exceeding the maximum number of steps (`Error::MaxSteps`),
///   or potential stiffness detected by the solver (`Error::Stiffness`).
///
pub fn solve_dde<const L: usize, T, V, D, S, F, H, O>(
    solver: &mut S,
    dde: &F,
    t0: T,
    tf: T,
    y0: &V,
    phi: H,
    solout: &mut O,
) -> Result<Solution<T, V, D>, Error<T, V>>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: DDE<L, T, V, D>,
    H: Fn(T) -> V,
    S: NumericalMethod<L, T, V, H, D> + Interpolation<T, V>,
    O: Solout<T, V, D>,
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
    match solver.init(dde, t0, tf, y0, phi) {
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

    // For event detection
    let mut tc: T = t0;
    let mut ts: T;

    // Check Terminate event at the start
    match dde.event(t0, y0) {
        ControlFlag::Continue => {}
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
            solver.set_h(tf - solver.t());
            solving = false;
        }

        // Perform a step
        solution.steps += 1;
        match solver.step(dde) {
            Ok(evals) => {
                solution.evals += evals;

                if let Status::RejectedStep = solver.status() {
                    solution.rejected_steps += 1;
                    continue;
                } else {
                    solution.accepted_steps += 1;
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
            ControlFlag::Terminate(re) => {
                let mut reason = re;
                ts = solver.t();

                let mut side_count = 0;
                let mut f_low: T = T::from_f64(-1.0).unwrap();
                let mut f_high: T = T::from_f64(1.0).unwrap();
                let mut t_guess: T;

                let max_iterations = 20;
                let tol = T::from_f64(1e-10).unwrap();

                for _ in 0..max_iterations {
                    if (ts - tc).abs() <= tol {
                        break;
                    }

                    t_guess = (tc * f_high - ts * f_low) / (f_high - f_low);

                    if !t_guess.is_finite()
                        || (integration_direction > T::zero() && (t_guess <= tc || t_guess >= ts))
                        || (integration_direction < T::zero() && (t_guess >= tc || t_guess <= ts))
                    {
                        t_guess = (tc + ts) / T::from_f64(2.0).unwrap();
                    }

                    let y = solver.interpolate(t_guess).unwrap();

                    match dde.event(t_guess, &y) {
                        ControlFlag::Continue => {
                            tc = t_guess;
                            side_count += 1;
                            if side_count >= 2 {
                                f_high /= T::from_f64(2.0).unwrap();
                                side_count = 0;
                            }
                        }
                        ControlFlag::Terminate(re) => {
                            reason = re;
                            ts = t_guess;
                            side_count = 0;
                            f_low = T::from_f64(-1.0).unwrap();
                        }
                    }
                }

                let y_final = solver.interpolate(ts).unwrap();

                let cutoff_index = if integration_direction > T::zero() {
                    solution.t.iter().position(|&t| t > ts)
                } else {
                    solution.t.iter().position(|&t| t < ts)
                };

                if let Some(idx) = cutoff_index {
                    solution.truncate(idx);
                }

                solution.push(ts, y_final);

                solution.status = Status::Interrupted(reason);
                solution.timer.complete();

                return Ok(solution);
            }
        }
    }

    solver.set_status(Status::Complete);

    solution.status = Status::Complete;
    solution.timer.complete();

    Ok(solution)
}
