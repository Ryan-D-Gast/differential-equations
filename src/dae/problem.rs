//! Initial Value Problem Struct and Constructors

use crate::{
    dae::{AlgebraicNumericalMethod, DAE, solve_dae},
    error::Error,
    interpolate::Interpolation,
    solout::*,
    solution::Solution,
    traits::{Real, State},
};

/// Initial Value Problem for Differential Algebraic Equations (DAEs)
///
/// The Initial Value Problem takes the form:
/// m·y' = f(t, y), a <= t <= b, y(a) = alpha
///
/// # Overview
///
/// The DAEProblem struct provides a simple interface for solving differential algebraic equations:
///
/// # Fields
///
/// * `dae` - DAE implementing the differential algebraic equation
/// * `t0` - Initial time
/// * `tf` - Final time
/// * `y0` - Initial state vector
///
/// # Basic Usage
///
/// * `new(dae, t0, tf, y0)` - Create a new DAEProblem
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
#[derive(Clone, Debug)]
pub struct DAEProblem<T, Y, F>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
{
    // Initial Value Problem Fields
    pub dae: F, // DAE containing the Differential Algebraic Equation and Optional Terminate Function.
    pub t0: T,  // Initial Time.
    pub tf: T,  // Final Time.
    pub y0: Y,  // Initial State Vector.
}

impl<T, Y, F> DAEProblem<T, Y, F>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
{
    /// Create a new Initial Value Problem
    ///
    /// # Arguments
    /// * `dae`  - DAE containing the Differential Algebraic Equation and Optional Terminate Function.
    /// * `t0`      - Initial Time.
    /// * `tf`      - Final Time.
    /// * `y0`      - Initial State Vector.
    ///
    /// # Returns
    /// * DAEProblem Problem ready to be solved.
    ///
    pub fn new(dae: F, t0: T, tf: T, y0: Y) -> Self {
        DAEProblem { dae, t0, tf, y0 }
    }

    /// Solve the DAEProblem using a default solout, e.g. outputting solutions at calculated steps.
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Status<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Status)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: AlgebraicNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        let mut default_solout = DefaultSolout::new(); // Default solout implementation
        solve_dae(
            solver,
            &self.dae,
            self.t0,
            self.tf,
            &self.y0,
            &mut default_solout,
        )
    }

    /// Returns a DAEProblem AlgebraicNumericalMethod with the provided solout function for outputting points.
    ///
    /// # Returns
    /// * DAEProblem AlgebraicNumericalMethod with the provided solout function ready for .solve() method.
    ///
    pub fn solout<'a, O: Solout<T, Y>>(
        &'a self,
        solout: &'a mut O,
    ) -> DAEProblemMutRefSoloutPair<'a, T, Y, F, O> {
        DAEProblemMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point.
    ///
    /// # Returns
    /// * DAEProblem AlgebraicNumericalMethod with Even Solout function ready for .solve() method.
    ///
    pub fn even(&self, dt: T) -> DAEProblemSoloutPair<'_, T, Y, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf); // Even solout implementation
        DAEProblemSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    ///
    /// # Returns
    /// * DAEProblem AlgebraicNumericalMethod with Dense Output function ready for .solve() method.
    ///
    pub fn dense(&self, n: usize) -> DAEProblemSoloutPair<'_, T, Y, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n); // Dense solout implementation
        DAEProblemSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `points` - Custom output points.
    ///
    /// # Returns
    /// * DAEProblem AlgebraicNumericalMethod with Custom Time Evaluation function ready for .solve() method.
    ///
    pub fn t_eval(&self, points: Vec<T>) -> DAEProblemSoloutPair<'_, T, Y, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf); // Custom time evaluation solout implementation
        DAEProblemSoloutPair::new(self, t_eval_solout)
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
    /// * DAEProblem AlgebraicNumericalMethod with CrossingSolout function ready for .solve() method.
    ///
    pub fn crossing(
        &self,
        component_idx: usize,
        threshhold: T,
        direction: CrossingDirection,
    ) -> DAEProblemSoloutPair<'_, T, Y, F, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshhold).with_direction(direction); // Crossing solout implementation
        DAEProblemSoloutPair::new(self, crossing_solout)
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
    /// * DAEProblem AlgebraicNumericalMethod with HyperplaneCrossingSolout function ready for .solve() method.
    ///
    pub fn hyperplane_crossing<V1>(
        &self,
        point: V1,
        normal: V1,
        extractor: fn(&Y) -> V1,
        direction: CrossingDirection,
    ) -> DAEProblemSoloutPair<'_, T, Y, F, HyperplaneCrossingSolout<T, V1, Y>>
    where
        V1: State<T>,
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        DAEProblemSoloutPair::new(self, solout)
    }

    /// Uses an `EventSolout` to detect zero crossings g(t,y)=0 for a user-defined event.
    pub fn event<'e, E>(
        &'e self,
        event: &'e E,
    ) -> DAEProblemSoloutPair<'e, T, Y, F, EventSolout<'e, T, Y, E>>
    where
        E: Event<T, Y> + 'e,
    {
        let solout = EventSolout::new(event, self.t0, self.tf);
        DAEProblemSoloutPair::new(self, solout)
    }
}

/// DAEProblemMutRefSoloutPair serves as a intermediate between the DAEProblem struct and a custom solout provided by the user.
pub struct DAEProblemMutRefSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
    O: Solout<T, Y>,
{
    pub problem: &'a DAEProblem<T, Y, F>, // Reference to the DAEProblem struct
    pub solout: &'a mut O,                // Reference to the solout implementation
}

impl<'a, T, Y, F, O> DAEProblemMutRefSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
    O: Solout<T, Y>,
{
    /// Create a new DAEProblemMutRefSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the DAEProblem struct
    ///
    pub fn new(problem: &'a DAEProblem<T, Y, F>, solout: &'a mut O) -> Self {
        DAEProblemMutRefSoloutPair { problem, solout }
    }

    /// Solve the DAEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - AlgebraicNumericalMethod to use for solving the DAEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: AlgebraicNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        solve_dae(
            solver,
            &self.problem.dae,
            self.problem.t0,
            self.problem.tf,
            &self.problem.y0,
            self.solout,
        )
    }
}

/// DAEProblemSoloutPair serves as a intermediate between the DAEProblem struct and solve_dae when a predefined solout is used.
#[derive(Clone, Debug)]
pub struct DAEProblemSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
    O: Solout<T, Y>,
{
    pub problem: &'a DAEProblem<T, Y, F>, // Reference to the DAEProblem struct
    pub solout: O,                        // Solout implementation
}

impl<'a, T, Y, F, O> DAEProblemSoloutPair<'a, T, Y, F, O>
where
    T: Real,
    Y: State<T>,
    F: DAE<T, Y>,
    O: Solout<T, Y>,
{
    /// Create a new DAEProblemSoloutPair
    ///
    /// # Arguments
    /// * `problem` - Reference to the DAEProblem struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(problem: &'a DAEProblem<T, Y, F>, solout: O) -> Self {
        DAEProblemSoloutPair { problem, solout }
    }

    /// Solve the DAEProblem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - AlgebraicNumericalMethod to use for solving the DAEProblem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        S: AlgebraicNumericalMethod<T, Y> + Interpolation<T, Y>,
    {
        solve_dae(
            solver,
            &self.problem.dae,
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
    ) -> DAEProblemSoloutPair<'a, T, Y, F, EventWrappedSolout<'a, T, Y, O, E>>
    where
        E: Event<T, Y>,
    {
        let wrapped = EventWrappedSolout::new(self.solout, event, self.problem.t0, self.problem.tf);
        DAEProblemSoloutPair {
            problem: self.problem,
            solout: wrapped,
        }
    }
}
