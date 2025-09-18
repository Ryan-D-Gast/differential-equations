//! Initial Value Problem Struct and Constructors

use crate::{
    error::Error,
    interpolate::Interpolation,
    ode::{ODE, OrdinaryNumericalMethod, solve_ode},
    solout::*,
    solution::Solution,
    traits::{Real, State},
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
/// use differential_equations::prelude::*;
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
/// let mut solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-6);
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
/// use differential_equations::prelude::*;
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
/// let mut method = ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12);
///
/// // Basic usage with default output points
/// let problem = ODEProblem::new(ode, 0.0, 10.0, vector![1.0, 0.0]);
/// let results = problem.solve(&mut method).unwrap();
///
/// // Advanced: evenly spaced output with 0.1 time intervals
/// let results = problem.dense(4).solve(&mut method).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ODEProblem<'a, T, Y, F>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
{
    /// ODE object implementing [`ODE`](crate::ode::ODE) trait
    pub ode: &'a F,
    /// Initial Time
    pub t0: T,
    /// Final Time
    pub tf: T,
    /// Initial State Vector
    pub y0: Y,
}

impl<'a, T, Y, F> ODEProblem<'a, T, Y, F>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
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
    pub fn new(ode: &'a F, t0: T, tf: T, y0: Y) -> Self {
        ODEProblem { ode, t0, tf, y0 }
    }

    /// Solve the ODEProblem using a default solout, e.g. outputting solutions at calculated steps.
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Status<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Status)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        let mut default_solout = DefaultSolout::new();
        solve_ode(
            solver,
            self.ode,
            self.t0,
            self.tf,
            &self.y0,
            &mut default_solout,
        )
    }

    /// Returns an ODEProblem OrdinaryNumericalMethod with the provided solout function for outputting points.
    ///
    /// # Returns
    /// * ODEProblem OrdinaryNumericalMethod with the provided solout function ready for .solve() method.
    ///
    pub fn solout<O: Solout<T, Y>>(
        &'a self,
        solout: &'a mut O,
    ) -> ODEProblemMutRefSoloutPair<'a, T, Y, F, O> {
        ODEProblemMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point.
    ///
    /// # Returns
    /// * ODEProblem OrdinaryNumericalMethod with Even Solout function ready for .solve() method.
    ///
    pub fn even(&self, dt: T) -> ODEProblemSoloutPair<'_, T, Y, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf);
        ODEProblemSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    ///
    /// # Returns
    /// * ODEProblem OrdinaryNumericalMethod with Dense Output function ready for .solve() method.
    ///
    pub fn dense(&self, n: usize) -> ODEProblemSoloutPair<'_, T, Y, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n);
        ODEProblemSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `points` - Custom output points.
    ///
    /// # Returns
    /// * ODEProblem OrdinaryNumericalMethod with Custom Time Evaluation function ready for .solve() method.
    ///
    pub fn t_eval(
        &self,
        points: impl AsRef<[T]>,
    ) -> ODEProblemSoloutPair<'_, T, Y, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf);
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
    /// * ODEProblem OrdinaryNumericalMethod with CrossingSolout function ready for .solve() method.
    ///
    pub fn crossing(
        &self,
        component_idx: usize,
        threshhold: T,
        direction: CrossingDirection,
    ) -> ODEProblemSoloutPair<'_, T, Y, F, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshhold).with_direction(direction);
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
    /// * ODEProblem OrdinaryNumericalMethod with HyperplaneCrossingSolout function ready for .solve() method.
    ///
    pub fn hyperplane_crossing<Y1>(
        &self,
        point: Y1,
        normal: Y1,
        extractor: fn(&Y) -> Y1,
        direction: CrossingDirection,
    ) -> ODEProblemSoloutPair<'_, T, Y, F, HyperplaneCrossingSolout<T, Y1, Y>>
    where
        Y1: State<T>,
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        ODEProblemSoloutPair::new(self, solout)
    }

    /// Uses an `EventSolout` to detect zero crossings of a user-defined event function (SciPy style).
    /// The provided event implements the `Event` trait returning a scalar function g(t,y) whose
    /// roots are sought. Each detected event point (t*, y*) is appended to the solution. Optional
    /// termination after N events can be configured in the Event implementation via `config()`.
    ///
    /// # Arguments
    /// * `event` - Object implementing `Event<T, Y>` whose zero crossings are desired.
    ///
    /// # Returns
    /// * `ODEProblemSoloutPair` with `EventSolout` ready for `.solve(&mut solver)`.
    ///
    /// # Example
    /// ```
    /// use differential_equations::prelude::*;
    /// use nalgebra::{Vector2, vector};
    ///
    /// struct SHO; // Simple harmonic oscillator
    /// impl ODE<f64, Vector2<f64>> for SHO {
    ///     fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
    ///         dydt[0]=y[1];
    ///         dydt[1]=-y[0];
    ///     }
    /// }
    ///
    /// // Event: detect when position crosses zero going positive (like SciPy event)
    /// struct ZeroUp;
    /// impl Event<f64, Vector2<f64>> for ZeroUp {
    ///     fn config(&self) -> EventConfig {
    ///         // Force only positive crossings
    ///         EventConfig::default().direction(CrossingDirection::Positive)
    ///     }
    /// 
    ///     fn event(&self, _t: f64, y: &Vector2<f64>) -> f64 {
    ///         y[0]
    ///     }
    /// }
    ///
    /// let osc = SHO; 
    /// let t0 = 0.0; 
    /// let tf = 10.0; 
    /// let y0 = vector![1.0, 0.0];
    /// let problem = ODEProblem::new(&osc, t0, tf, y0);
    /// let mut solver = ExplicitRungeKutta::dop853();
    /// let solution = problem.event(&ZeroUp).solve(&mut solver).unwrap();
    /// // solution.t now contains zero-up crossing times
    /// ```
    pub fn event<E>(&self, event: &'a E) -> ODEProblemSoloutPair<'_, T, Y, F, EventSolout<T, Y, E>>
    where
        E: Event<T, Y>,
    {
        let solout = EventSolout::new(event, self.t0, self.tf);
        ODEProblemSoloutPair::new(self, solout)
    }
}

/// ODEProblemMutRefSoloutPair serves as a intermediate between the ODEProblem struct and a custom solout provided by the user.
pub struct ODEProblemMutRefSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
    O: Solout<T, Y>,
{
    pub problem: &'a ODEProblem<'a, T, Y, F>,
    pub solout: &'a mut O,
}

impl<'a, T, Y, F, O> ODEProblemMutRefSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
    O: Solout<T, Y>,
{
    /// Create a new ODEProblemMutRefSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the ODEProblem struct
    ///
    pub fn new(problem: &'a ODEProblem<T, Y, F>, solout: &'a mut O) -> Self {
        ODEProblemMutRefSoloutPair { problem, solout }
    }

    /// Solve the ODEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - OrdinaryNumericalMethod to use for solving the ODEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        solve_ode(
            solver,
            self.problem.ode,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            self.solout,
        )
    }
}

/// ODEProblemSoloutPair serves as a intermediate between the ODEProblem struct and solve_ode when a predefined solout is used.
#[derive(Clone, Debug)]
pub struct ODEProblemSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
    O: Solout<T, Y>,
{
    pub problem: &'a ODEProblem<'a, T, Y, F>,
    pub solout: O,
}

impl<'a, T, Y, F, O> ODEProblemSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y>,
    O: Solout<T, Y>,
{
    /// Create a new ODEProblemSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the ODEProblem struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(problem: &'a ODEProblem<T, Y, F>, solout: O) -> Self {
        ODEProblemSoloutPair { problem, solout }
    }

    /// Solve the ODEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - OrdinaryNumericalMethod to use for solving the ODEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        solve_ode(
            solver,
            self.problem.ode,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            &mut self.solout,
        )
    }

    /// Wrap current solout with event detection while preserving original output strategy.
    pub fn event<E>(
        self,
        event: &'a E,
    ) -> ODEProblemSoloutPair<'a, T, Y, F, EventWrappedSolout<'a, T, Y, O, E>>
    where
        E: Event<T, Y>,
    {
        let wrapped = EventWrappedSolout::new(self.solout, event, self.problem.t0, self.problem.tf);
        ODEProblemSoloutPair {
            problem: self.problem,
            solout: wrapped,
        }
    }
}
