//! Default Solout Implementation, e.g. outputting solutions at calculated steps.
//!
//! This module provides the default output strategy, which returns solution points
//! at each step taken by the solver without any interpolation.

use super::*;

/// The default output handler that returns solution values at each solver step.
///
/// # Overview
///
/// `DefaultSolout` is the simplest output handler that captures the solution
/// at each internal step calculated by the solver. It doesn't perform any
/// interpolation or filtering - it simply records the exact points that the
/// solver naturally computes during integration.
///
/// # Features
///
/// - Captures all solver steps in the output
/// - No interpolation overhead
/// - Gives the raw, unmodified solver trajectory
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
/// use differential_equations::solout::DefaultSolout;
/// use nalgebra::{Vector1, vector};
///
/// // Simple exponential growth
/// struct ExponentialGrowth;
///
/// impl ODE<f64, Vector1<f64>> for ExponentialGrowth {
///     fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
///         dydt[0] = y[0]; // dy/dt = y
///     }
/// }
///
/// // Create the system and solver
/// let system = ExponentialGrowth;
/// let t0 = 0.0;
/// let tf = 2.0;
/// let y0 = vector![1.0];
/// let mut solver = DOP853::new().rtol(1e-6).atol(1e-8);
///
/// // Use the default output handler explicitly
/// let mut default_output = DefaultSolout::new();
///
/// // Solve with default output
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solout(&mut default_output).solve(&mut solver).unwrap();
///
/// // Note: This is equivalent to the default behavior
/// let solution2 = ivp.solve(&mut solver).unwrap();
/// ```
///
/// # Output Characteristics
///
/// The output will contain only the actual steps computed by the solver,
/// which may not be evenly spaced in time. The spacing depends on the solver's
/// adaptive step size control.
///
/// For evenly spaced output points, consider using `EvenSolout` instead.
///
pub struct DefaultSolout {}

impl<T, V, D> Solout<T, V, D> for DefaultSolout
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    fn solout<I>(
        &mut self,
        t_curr: T,
        _t_prev: T,
        y_curr: &V,
        _y_prev: &V,
        _interpolator: &mut I,
        solution: &mut Solution<T, V, D>,
    ) -> ControlFlag<D>
    where
        I: Interpolation<T, V>,
    {
        // Output the current time and state to the vectors
        solution.push(t_curr, *y_curr);

        // Continue the integration
        ControlFlag::Continue
    }
}

impl Default for DefaultSolout {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultSolout {
    /// Creates a new DefaultSolout instance.
    ///
    /// This is the simplest output handler that captures solution values
    /// at each step naturally taken by the solver.
    ///
    /// # Returns
    /// * A new `DefaultSolout` instance
    ///
    pub fn new() -> Self {
        DefaultSolout {}
    }
}
