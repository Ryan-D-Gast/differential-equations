//! Initial Value Problem Struct and Constructors for Delay Differential Equations (DDEs)

use crate::{
    Error, Solution,
    dde::{DDE, numerical_method::DelayNumericalMethod, solve_dde},
    interpolate::Interpolation,
    solout::*,
    traits::{CallBackData, Real, State},
};
use std::marker::PhantomData;

/// Initial Value Problem for Delay Differential Equations (DDEs)
///
/// The Initial Value Problem takes the form:
/// y'(t) = f(t, y(t), y(t-tau1), ...),  t ∈ [t0, tf],  y(t) = phi(t) for t <= t0
///
/// # Overview
///
/// The DDEProblem struct provides a simple interface for solving DDEs:
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
/// use nalgebra::{Vector2, vector};
///
/// let mut rkf45 = ExplicitRungeKutta::rkf45()
///    .rtol(1e-6)
///    .atol(1e-6);
///
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// let phi = |t| {
///     y0
/// };
/// struct Example;
/// impl DDE<1, f64, Vector2<f64>> for Example {
///     fn diff(&self, t: f64, y: &Vector2<f64>, yd: &[Vector2<f64>; 1], dydt: &mut Vector2<f64>) {
///        dydt[0] = y[1] + yd[0][0];
///        dydt[1] = -y[0] + yd[0][1];
///     }
///
///     fn lags(&self, _t: f64, _y: &Vector2<f64>, lags: &mut [f64; 1]) {
///        lags[0] = 1.0; // Fixed delay of 1.0
///     }
/// }
/// let solution = DDEProblem::new(Example, t0, tf, y0, phi).solve(&mut rkf45).unwrap();
///
/// let (t, y) = solution.last().unwrap();
/// println!("Solution: ({}, {})", t, y);
/// ```
///
/// # Fields
///
/// * `dde` - DDE implementing the differential equation
/// * `t0` - Initial time
/// * `tf` - Final time
/// * `y0` - Initial state vector at `t0`
/// * `phi` - Initial history function `phi(t)` for `t <= t0`
///
/// # Basic Usage
///
/// * `new(dde, t0, tf, y0, phi)` - Create a new DDEProblem
/// * `solve(&mut solver)` - Solve using default output (solver step points)
///
/// # Output Control Methods
///
/// These methods configure how solution points are generated and returned:
///
/// * `even(dt)` - Generate evenly spaced output points with interval `dt`
/// * `dense(n)` - Include `n` interpolated points between each solver step
/// * `t_eval(points)` - Evaluate solution at specific time points
/// * `solout(custom_solout)` - Use a custom output handler
///
/// Each returns a solver configuration that can be executed with `.solve(&mut solver)`.
#[derive(Clone, Debug)] // Note: Clone requires F and H to be Clone
pub struct DDEProblem<const L: usize, T, Y, D, F, H>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y,
{
    // Initial Value Problem Fields
    pub dde: F, // DDE containing the Differential Equation and Optional Terminate Function.
    pub t0: T,  // Initial Time.
    pub tf: T,  // Final Time.
    pub y0: Y,  // Initial State Vector.
    pub phi: H, // Initial History Function.

    // Phantom Data for Users event output
    _event_output_type: PhantomData<D>,
}

impl<const L: usize, T, Y, D, F, H> DDEProblem<L, T, Y, D, F, H>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y + Clone, // Require Clone for H if DDEProblem needs Clone
{
    /// Create a new Initial Value Problem for DDEs
    ///
    /// # Arguments
    /// * `dde`  - DDE containing the Differential Equation and Optional Terminate Function.
    /// * `t0`   - Initial Time.
    /// * `tf`   - Final Time.
    /// * `y0`   - Initial State Vector at `t0`.
    /// * `phi`  - Initial History Function `phi(t)` for `t <= t0`.
    ///
    /// # Returns
    /// * DDEProblem Problem ready to be solved.
    ///
    pub fn new(dde: F, t0: T, tf: T, y0: Y, phi: H) -> Self {
        DDEProblem {
            dde,
            t0,
            tf,
            y0,
            phi,
            _event_output_type: PhantomData,
        }
    }

    /// Solve the DDEProblem using a default solout, e.g. outputting solutions at calculated steps.
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered.
    ///
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: DelayNumericalMethod<L, T, Y, H, D> + Interpolation<T, Y>,
        H: Clone, // phi needs to be cloneable for solve_dde
    {
        let mut default_solout = DefaultSolout::new(); // Default solout implementation
        solve_dde(
            solver,
            &self.dde,
            self.t0,
            self.tf,
            &self.y0,
            self.phi.clone(),
            &mut default_solout,
        )
    }

    /// Returns a DDEProblem DelayNumericalMethod with the provided solout function for outputting points.
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with the provided solout function ready for .solve() method.
    ///
    pub fn solout<'a, O: Solout<T, Y, D>>(
        &'a self,
        solout: &'a mut O,
    ) -> DDEProblemMutRefSoloutPair<'a, L, T, Y, D, F, H, O> {
        DDEProblemMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point.
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with Even Solout function ready for .solve() method.
    ///
    pub fn even(&self, dt: T) -> DDEProblemSoloutPair<'_, L, T, Y, D, F, H, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf); // Even solout implementation
        DDEProblemSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with Dense Output function ready for .solve() method.
    ///
    pub fn dense(&self, n: usize) -> DDEProblemSoloutPair<'_, L, T, Y, D, F, H, DenseSolout> {
        let dense_solout = DenseSolout::new(n); // Dense solout implementation
        DDEProblemSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `points` - Custom output points.
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with Custom Time Evaluation function ready for .solve() method.
    ///
    pub fn t_eval(
        &self,
        points: Vec<T>,
    ) -> DDEProblemSoloutPair<'_, L, T, Y, D, F, H, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf); // Custom time evaluation solout implementation
        DDEProblemSoloutPair::new(self, t_eval_solout)
    }

    /// Uses the CrossingSolout method to output points when a specific component crosses a threshold.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `component_idx` - Index of the component to monitor for crossing.
    /// * `threshhold` - Value to cross.
    /// * `direction` - Direction of crossing (positive or negative).
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with CrossingSolout function ready for .solve() method.
    ///
    pub fn crossing(
        &self,
        component_idx: usize,
        threshhold: T,
        direction: CrossingDirection,
    ) -> DDEProblemSoloutPair<'_, L, T, Y, D, F, H, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshhold).with_direction(direction); // Crossing solout implementation
        DDEProblemSoloutPair::new(self, crossing_solout)
    }

    /// Uses the HyperplaneCrossingSolout method to output points when a specific hyperplane is crossed.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `point` - Point on the hyperplane.
    /// * `normal` - Normal vector of the hyperplane.
    /// * `extractor` - Function to extract the component from the state vector.
    /// * `direction` - Direction of crossing (positive or negative).
    ///
    /// # Returns
    /// * DDEProblem DelayNumericalMethod with HyperplaneCrossingSolout function ready for .solve() method.
    ///
    pub fn hyperplane_crossing<Y1>(
        &self,
        point: Y1,
        normal: Y1,
        extractor: fn(&Y) -> Y1,
        direction: CrossingDirection,
    ) -> DDEProblemSoloutPair<'_, L, T, Y, D, F, H, HyperplaneCrossingSolout<T, Y1, Y>>
    where
        Y1: State<T>,
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        DDEProblemSoloutPair::new(self, solout)
    }
}

/// DDEProblemMutRefSoloutPair serves as an intermediate between the DDEProblem struct and a custom solout provided by the user.
pub struct DDEProblemMutRefSoloutPair<'a, const L: usize, T, Y, D, F, H, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y,
    O: Solout<T, Y, D>,
{
    pub problem: &'a DDEProblem<L, T, Y, D, F, H>, // Reference to the DDEProblem struct
    pub solout: &'a mut O,                         // Reference to the solout implementation
}

impl<'a, const L: usize, T, Y, D, F, H, O> DDEProblemMutRefSoloutPair<'a, L, T, Y, D, F, H, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y + Clone, // Require Clone for H
    O: Solout<T, Y, D>,
{
    /// Create a new DDEProblemMutRefSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the DDEProblem struct
    /// * `solout` - Mutable reference to the solout implementation
    ///
    pub fn new(problem: &'a DDEProblem<L, T, Y, D, F, H>, solout: &'a mut O) -> Self {
        DDEProblemMutRefSoloutPair { problem, solout }
    }

    /// Solve the DDEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - DelayNumericalMethod to use for solving the DDEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered.
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: DelayNumericalMethod<L, T, Y, H, D> + Interpolation<T, Y>,
    {
        solve_dde(
            solver,
            &self.problem.dde,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            self.problem.phi.clone(),
            self.solout,
        )
    }
}

/// DDEProblemSoloutPair serves as an intermediate between the DDEProblem struct and solve_dde when a predefined solout is used.
#[derive(Clone, Debug)]
pub struct DDEProblemSoloutPair<'a, const L: usize, T, Y, D, F, H, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y,
    O: Solout<T, Y, D>,
{
    pub problem: &'a DDEProblem<L, T, Y, D, F, H>,
    pub solout: O,
}

impl<'a, const L: usize, T, Y, D, F, H, O> DDEProblemSoloutPair<'a, L, T, Y, D, F, H, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: DDE<L, T, Y, D>,
    H: Fn(T) -> Y + Clone,
    O: Solout<T, Y, D>,
{
    /// Create a new DDEProblemSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the DDEProblem struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(problem: &'a DDEProblem<L, T, Y, D, F, H>, solout: O) -> Self {
        DDEProblemSoloutPair { problem, solout }
    }

    /// Solve the DDEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - DelayNumericalMethod to use for solving the DDEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered.
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: DelayNumericalMethod<L, T, Y, H, D> + Interpolation<T, Y>,
    {
        solve_dde(
            solver,
            &self.problem.dde,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            self.problem.phi.clone(),
            &mut self.solout,
        )
    }
}
