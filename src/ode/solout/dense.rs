//! Dense Solout Implementation, for outputting a dense set of points.
//!
//! This module provides an output strategy that generates additional interpolated
//! points between each solver step, creating a denser output representation.

use super::*;

/// An output handler that provides a dense set of interpolated points between solver steps.
/// 
/// # Overview
///
/// `DenseSolout` enhances the solution output by interpolating additional points
/// between the naturally computed solver steps. This creates a smoother, more
/// detailed trajectory that can better represent the continuous solution,
/// especially when the solver takes large steps.
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
/// use differential_equations::ode::solout::DenseSolout;
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
/// // Generate 9 additional points between each solver step (10 total per interval)
/// let mut dense_output = DenseSolout::new(10);
///
/// // Solve with dense output
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solout(&mut dense_output).solve(&mut solver).unwrap();
///
/// // Note: This is equivalent to using the convenience method:
/// let solution = ivp.dense(10).solve(&mut solver).unwrap();
/// ```
///
/// # Output Characteristics
///
/// The output will contain both the original solver steps and additional interpolated
/// points between them. The interpolated points are evenly spaced within each step.
/// 
/// For example, with n=5:
/// - Original solver steps: t₀, t₁, t₂, ...
/// - Dense output: t₀, t₀+h/5, t₀+2h/5, t₀+3h/5, t₀+4h/5, t₁, t₁+h/5, ...
///
/// # Performance Considerations
///
/// Increasing the number of interpolation points increases computational cost and
/// memory usage. Choose a value that balances the need for smooth output with
/// performance requirements.
/// 
pub struct DenseSolout {
    /// Number of points between steps (including the endpoints)
    n: usize,
}

impl<T, const R: usize, const C: usize, E> Solout<T, R, C, E> for DenseSolout
where 
    T: Real,
    E: EventData
{
    fn solout<S>(&mut self, solver: &mut S, solution: &mut Solution<T, R, C, E>)
    where 
        S: Solver<T, R, C, E> 
    {
        let t_prev = solver.t_prev();
        let t_curr = solver.t();
        
        // Interpolate between steps
        for i in 1..self.n {
            let h_old = t_curr - t_prev;
            let ti = t_prev + T::from_usize(i).unwrap() * h_old / T::from_usize(self.n).unwrap();
            let yi = solver.interpolate(ti).unwrap();
            solution.push(ti, yi);
        }

        // Save actual calculated step as well
        solution.push(t_curr, *solver.y());
    }
}

impl DenseSolout {
    /// Creates a new DenseSolout instance with the specified number of points per interval.
    ///
    /// # Arguments
    /// * `n` - Number of points per interval, including endpoints. For example, n=5 will 
    ///         add 4 interpolated points between each solver step, plus the solver step itself.
    ///
    /// # Returns
    /// * A new `DenseSolout` instance
    /// 
    pub fn new(n: usize) -> Self {
        DenseSolout { n }
    }
}
