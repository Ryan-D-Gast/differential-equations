//! Initial Value Problem Struct and Constructors

use crate::{
    Error, Solution,
    interpolate::Interpolation,
    ode::{ODE, solve::solve_problem, numerical_method::NumericalMethod},
    solout::*,
    traits::{CallBackData, Real, State},
};

/// Initial Value Problem for Ordinary Differential Equations (ODEs)
///
/// The Initial Value Problem takes the form:
/// y' = f(t, y), a <= t <= b, y(a) = alpha
///
/// # Overview
///
/// The ODEProblem struct provides a simple interface for solving differential equations:
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
///
/// struct LinearEquation {
///    pub a: f32,
///    pub b: f32,
/// }
///
/// impl ODE<f32, f32> for LinearEquation {
///    fn diff(&self, _t: f32, y: &f32, dydt: &mut f32) {
///        *dydt = self.a + self.b * y;
///   }
/// }
///
/// // Create the ode and initial conditions
/// let ode = LinearEquation { a: 1.0, b: 2.0 };
/// let t0 = 0.0;
/// let tf = 1.0;
/// let y0 = 1.0;
/// let mut solver = DOP853::new().rtol(1e-8).atol(1e-6);
///
/// // Basic usage:
/// let problem = ODEProblem::new(ode, t0, tf, y0);
/// let solution = problem.solve(&mut solver).unwrap();
///
/// // Advanced output control:
/// let solution = problem.even(0.1).solve(&mut solver).unwrap();
/// ```
///
/// # Fields
///
/// * `ode` - ODE implementing the differential equation
/// * `t0` - Initial time
/// * `tf` - Final time
/// * `y0` - Initial state vector
///
/// # Basic Usage
///
/// * `new(ode, t0, tf, y0)` - Create a new ODEProblem
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
///
/// # Example 2
///
/// ```
/// use differential_equations::ode::*;
/// use nalgebra::{SVector, vector};
///
/// struct HarmonicOscillator { k: f64 }
///
/// impl ODE<f64, SVector<f64, 2>> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///         dydt[0] = y[1];
///         dydt[1] = -self.k * y[0];
///     }
/// }
///
/// let ode = HarmonicOscillator { k: 1.0 };
/// let mut method = DOP853::new().rtol(1e-12).atol(1e-12);
///
/// // Basic usage with default output points
/// let problem = ODEProblem::new(ode, 0.0, 10.0, vector![1.0, 0.0]);
/// let results = problem.solve(&mut method).unwrap();
///
/// // Advanced: evenly spaced output with 0.1 time intervals
/// let results = problem.dense(4).solve(&mut method).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ODEProblem<T, V, D, F>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
{
    // Initial Value Problem Fields
    pub ode: F, // ODE containing the Differential Equation and Optional Terminate Function.
    pub t0: T,  // Initial Time.
    pub tf: T,  // Final Time.
    pub y0: V,  // Initial State Vector.

    // Phantom Data for Users event output
    _event_output_type: std::marker::PhantomData<D>,
}

impl<T, V, D, F> ODEProblem<T, V, D, F>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
{
    /// Create a new Initial Value Problem
    ///
    /// # Arguments
    /// * `ode`  - ODE containing the Differential Equation and Optional Terminate Function.
    /// * `t0`      - Initial Time.
    /// * `tf`      - Final Time.
    /// * `y0`      - Initial State Vector.
    ///
    /// # Returns
    /// * ODEProblem Problem ready to be solved.
    ///
    pub fn new(ode: F, t0: T, tf: T, y0: V) -> Self {
        ODEProblem {
            ode,
            t0,
            tf,
            y0,
            _event_output_type: std::marker::PhantomData,
        }
    }

    /// Solve the ODEProblem using a default solout, e.g. outputting solutions at calculated steps.
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Status<T, V, D>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Status)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        let mut default_solout = DefaultSolout::new(); // Default solout implementation
        solve_problem(
            solver,
            &self.ode,
            self.t0,
            self.tf,
            &self.y0,
            &mut default_solout,
        )
    }

    /// Returns an ODEProblem NumericalMethod with the provided solout function for outputting points.
    ///
    /// # Returns
    /// * ODEProblem NumericalMethod with the provided solout function ready for .solve() method.
    ///
    pub fn solout<'a, O: Solout<T, V, D>>(
        &'a self,
        solout: &'a mut O,
    ) -> ODEProblemMutRefSoloutPair<'a, T, V, D, F, O> {
        ODEProblemMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point.
    ///
    /// # Returns
    /// * ODEProblem NumericalMethod with Even Solout function ready for .solve() method.
    ///
    pub fn even(&self, dt: T) -> ODEProblemSoloutPair<'_, T, V, D, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf); // Even solout implementation
        ODEProblemSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    ///
    /// # Returns
    /// * ODEProblem NumericalMethod with Dense Output function ready for .solve() method.
    ///
    pub fn dense(&self, n: usize) -> ODEProblemSoloutPair<'_, T, V, D, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n); // Dense solout implementation
        ODEProblemSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `points` - Custom output points.
    ///
    /// # Returns
    /// * ODEProblem NumericalMethod with Custom Time Evaluation function ready for .solve() method.
    ///
    pub fn t_eval(&self, points: Vec<T>) -> ODEProblemSoloutPair<'_, T, V, D, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf); // Custom time evaluation solout implementation
        ODEProblemSoloutPair::new(self, t_eval_solout)
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
    /// * ODEProblem NumericalMethod with CrossingSolout function ready for .solve() method.
    ///
    pub fn crossing(
        &self,
        component_idx: usize,
        threshhold: T,
        direction: CrossingDirection,
    ) -> ODEProblemSoloutPair<'_, T, V, D, F, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshhold).with_direction(direction); // Crossing solout implementation
        ODEProblemSoloutPair::new(self, crossing_solout)
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
    /// * ODEProblem NumericalMethod with HyperplaneCrossingSolout function ready for .solve() method.
    ///
    pub fn hyperplane_crossing<V1>(
        &self,
        point: V1,
        normal: V1,
        extractor: fn(&V) -> V1,
        direction: CrossingDirection,
    ) -> ODEProblemSoloutPair<'_, T, V, D, F, HyperplaneCrossingSolout<T, V1, V>>
    where
        V1: State<T>,
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        ODEProblemSoloutPair::new(self, solout)
    }
}

/// ODEProblemMutRefSoloutPair serves as a intermediate between the ODEProblem struct and a custom solout provided by the user.
pub struct ODEProblemMutRefSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    pub problem: &'a ODEProblem<T, V, D, F>, // Reference to the ODEProblem struct
    pub solout: &'a mut O,        // Reference to the solout implementation
}

impl<'a, T, V, D, F, O> ODEProblemMutRefSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    /// Create a new ODEProblemMutRefSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the ODEProblem struct
    ///
    pub fn new(problem: &'a ODEProblem<T, V, D, F>, solout: &'a mut O) -> Self {
        ODEProblemMutRefSoloutPair { problem, solout }
    }

    /// Solve the ODEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - NumericalMethod to use for solving the ODEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Error<T, V>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        solve_problem(
            solver,
            &self.problem.ode,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            self.solout,
        )
    }
}

/// ODEProblemSoloutPair serves as a intermediate between the ODEProblem struct and solve_problem when a predefined solout is used.
#[derive(Clone, Debug)]
pub struct ODEProblemSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    pub problem: &'a ODEProblem<T, V, D, F>, // Reference to the ODEProblem struct
    pub solout: O,                // Solout implementation
}

impl<'a, T, V, D, F, O> ODEProblemSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    /// Create a new ODEProblemSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the ODEProblem struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(problem: &'a ODEProblem<T, V, D, F>, solout: O) -> Self {
        ODEProblemSoloutPair { problem, solout }
    }

    /// Solve the ODEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - NumericalMethod to use for solving the ODEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Error<T, V>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        solve_problem(
            solver,
            &self.problem.ode,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            &mut self.solout,
        )
    }
}
