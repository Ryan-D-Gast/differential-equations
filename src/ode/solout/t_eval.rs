//! T-Evaluation Solout Implementation, for outputting at specific evaluation points.
//!
//! This module provides an output strategy that generates solution points at specific
//! user-defined time points through interpolation.

use super::*;

/// An output handler that evaluates the solution at specific user-defined time points.
/// 
/// # Overview
///
/// `TEvalSolout` provides the ability to evaluate the solution at a list of arbitrary 
/// time points specified by the user. This is useful when you need the solution at 
/// specific times that don't necessarily align with the solver's internal steps, 
/// such as for:
///
/// - Comparison with experimental data at specific measurement times
/// - Uniform or non-uniform time grids defined by the application
/// - Evaluating the solution at specific times of interest
///
/// The solout uses interpolation to evaluate the solution at each requested time point,
/// ensuring accurate results even if the solver doesn't naturally step through those points.
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
/// use differential_equations::ode::solout::TEvalSolout;
/// use nalgebra::{Vector2, vector};
///
/// // Simple harmonic oscillator
/// struct HarmonicOscillator;
///
/// impl ODE<f64, 2, 1> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
///         // y[0] = position, y[1] = velocity
///         dydt[0] = y[1];
///         dydt[1] = -y[0];
///     }
/// }
///
/// // Create the system and solver
/// let system = HarmonicOscillator;
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// let mut solver = DOP853::new().rtol(1e-6).atol(1e-8);
///
/// // Define specific time points of interest
/// let evaluation_points = vec![0.0, 0.5, 1.0, 2.0, 3.14, 5.0, 7.5, 10.0];
/// let mut t_eval_output = TEvalSolout::new(evaluation_points, t0, tf);
///
/// // Solve with specific evaluation points
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solout(&mut t_eval_output).solve(&mut solver).unwrap();
///
/// // Note: This is equivalent to using the convenience method:
/// let solution = ivp
///     .t_eval(vec![0.0, 0.5, 1.0, 2.0, 3.14, 5.0, 7.5, 10.0])
///     .solve(&mut solver).unwrap();
/// ```
///
/// # Output Characteristics
///
/// The output will contain exactly the points specified in the `t_evals` vector,
/// in the order appropriate for the integration direction. Points that fall outside
/// the integration interval [t0, tf] are ignored.
///
pub struct TEvalSolout<T: Real> {
    /// Points to evaluate the solution at
    t_evals: Vec<T>,
    /// Next evaluation point index
    next_eval_idx: usize,
    /// Direction of integration (positive for forward, negative for backward)
    integration_direction: T,
}

impl<T, const R: usize, const C: usize, E> Solout<T, R, C, E> for TEvalSolout<T>
where 
    T: Real,
    E: EventData
{
    fn solout<SV, SI>(&mut self, solver: &mut SV, solution: &mut SI)
    where 
        SV: Solver<T, R, C, E>,
        SI: SolutionInterface<T, R, C, E>
    {
        let t_prev = solver.t_prev();
        let t_curr = solver.t();
        
        // Process evaluation points that fall within current step
        let mut idx = self.next_eval_idx;
        while idx < self.t_evals.len() {
            let t_eval = self.t_evals[idx];
            
            // Check if this evaluation point is within the current step
            let in_range = if self.integration_direction > T::zero() {
                (t_eval == t_prev && idx == 0) || (t_eval > t_prev && t_eval <= t_curr)
            } else {
                (t_eval == t_prev && idx == 0) || (t_eval < t_prev && t_eval >= t_curr)
            };

            if in_range {
                // If the evaluation point is exactly at the current step, just use the solver's state
                if t_eval == t_curr {
                    solution.record(t_eval, *solver.y());
                } else {
                    // Otherwise interpolate
                    let y_eval = solver.interpolate(t_eval).unwrap();
                    solution.record(t_eval, y_eval);
                }
                idx += 1;
            } else {
                // If we've gone beyond the current step, stop processing
                if (self.integration_direction > T::zero() && t_eval > t_curr) ||
                   (self.integration_direction < T::zero() && t_eval < t_curr) {
                    break;
                }
                idx += 1;
            }
        }
        
        // Update next_eval_idx for the next call
        self.next_eval_idx = idx;
    }

    fn include_t0_tf(&self) -> bool {
        false // Do not include t0 and tf in the output
    }
}

impl<T: Real> TEvalSolout<T> {
    /// Creates a new TEvalSolout with the specified evaluation points.
    ///
    /// The evaluation points are automatically sorted according to the integration
    /// direction determined by `t0` and `tf`.
    ///
    /// # Arguments
    /// * `t_evals` - Vector of time points at which to evaluate the solution
    /// * `t0` - Initial time of the integration
    /// * `tf` - Final time of the integration
    ///
    /// # Returns
    /// * A new `TEvalSolout` instance
    ///
    pub fn new(t_evals: Vec<T>, t0: T, tf: T) -> Self {
        // Sort evaluation points according to integration direction
        let integration_direction = (tf - t0).signum();
        let mut sorted_t_evals = t_evals.clone();

        if integration_direction > T::zero() {
            sorted_t_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted_t_evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        TEvalSolout {
            t_evals: sorted_t_evals,
            next_eval_idx: 0,
            integration_direction,
        }
    }
}
