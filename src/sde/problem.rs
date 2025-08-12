//! SDE Problem Struct and Constructors

use crate::{
    error::Error,
    interpolate::Interpolation,
    sde::{SDE, StochasticNumericalMethod, solve_sde},
    solout::*,
    solution::Solution,
    traits::{CallBackData, Real, State},
};

/// Initial Value Problem for Stochastic Differential Equations (SDEProblem)
///
/// The Initial Value Problem takes the form:
/// dY = a(t, Y)dt + b(t, Y)dW, t0 <= t <= tf, Y(t0) = y0
///
/// where:
/// - a(t, Y) is the drift term (deterministic part)
/// - b(t, Y) is the diffusion term (stochastic part)
/// - dW represents a Wiener process increment
///
/// # Overview
///
/// The SDEProblem struct provides a simple interface for solving stochastic differential equations:
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
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
/// let mut solver = ExplicitRungeKutta::three_eighths(0.01);
/// let gbm = GBM::new(42);
/// let mut gbm_problem = SDEProblem::new(gbm, t0, tf, y0);
///
/// // Solve the SDE
/// let result = gbm_problem.solve(&mut solver);
/// ```
///
/// # Fields
///
/// * `sde` - SDE implementing the stochastic differential equation
/// * `t0` - Initial time
/// * `tf` - Final time
/// * `y0` - Initial state vector
///
/// # Basic Usage
///
/// * `new(sde, t0, tf, y0)` - Create a new SDE Problem
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
/// * `seed(u64)` - Set a specific random seed for reproducible simulations
///
#[derive(Clone, Debug)]
pub struct SDEProblem<T, Y, D, F>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
{
    // SDE Problem Fields
    pub sde: F, // SDE containing the Stochastic Differential Equation and Optional Terminate Function
    pub t0: T,  // Initial Time
    pub tf: T,  // Final Time
    pub y0: Y,  // Initial State Vector

    // Phantom Data for Users event output
    _event_output_type: std::marker::PhantomData<D>,
}

impl<T, Y, D, F> SDEProblem<T, Y, D, F>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
{
    /// Create a new Stochastic Differential Equation Problem
    ///
    /// # Arguments
    /// * `sde` - SDE containing the Stochastic Differential Equation and Optional Terminate Function
    /// * `t0` - Initial Time
    /// * `tf` - Final Time
    /// * `y0` - Initial State Vector
    ///
    /// # Returns
    /// * SDE Problem ready to be solved
    ///
    pub fn new(sde: F, t0: T, tf: T, y0: Y) -> Self {
        SDEProblem {
            sde,
            t0,
            tf,
            y0,
            _event_output_type: std::marker::PhantomData,
        }
    }

    /// Solve the SDE Problem using a default solout, e.g. outputting solutions at calculated steps
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: StochasticNumericalMethod<T, Y, D> + Interpolation<T, Y>,
    {
        let mut default_solout = DefaultSolout::new(); // Default solout implementation
        solve_sde(
            solver,
            &mut self.sde,
            self.t0,
            self.tf,
            &self.y0,
            &mut default_solout,
        )
    }

    /// Returns an SDE Problem with the provided solout function for outputting points
    ///
    /// # Returns
    /// * SDE Problem with the provided solout function ready for .solve() method
    ///
    pub fn solout<'a, O: Solout<T, Y, D>>(
        &'a mut self,
        solout: &'a mut O,
    ) -> SDEProblemMutRefSoloutPair<'a, T, Y, D, F, O> {
        SDEProblemMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time
    /// Note that this does not include the solution of the calculated steps
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point
    ///
    /// # Returns
    /// * SDE Problem with Even Solout function ready for .solve() method
    ///
    pub fn even(&mut self, dt: T) -> SDEProblemSoloutPair<'_, T, Y, D, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf);
        SDEProblemSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step
    /// Note this includes the solution of the calculated steps
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step
    ///
    /// # Returns
    /// * SDE Problem with Dense Output function ready for .solve() method
    ///
    pub fn dense(&mut self, n: usize) -> SDEProblemSoloutPair<'_, T, Y, D, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n);
        SDEProblemSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method
    /// Note this does not include the solution of the calculated steps
    ///
    /// # Arguments
    /// * `points` - Custom output points
    ///
    /// # Returns
    /// * SDE Problem with Custom Time Evaluation function ready for .solve() method
    ///
    pub fn t_eval(
        &mut self,
        points: impl AsRef<[T]>,
    ) -> SDEProblemSoloutPair<'_, T, Y, D, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf);
        SDEProblemSoloutPair::new(self, t_eval_solout)
    }

    /// Uses the CrossingSolout method to output points when a specific component crosses a threshold
    /// Note this does not include the solution of the calculated steps
    ///
    /// # Arguments
    /// * `component_idx` - Index of the component to monitor for crossing
    /// * `threshold` - Value to cross
    /// * `direction` - Direction of crossing (positive or negative)
    ///
    /// # Returns
    /// * SDE Problem with CrossingSolout function ready for .solve() method
    ///
    pub fn crossing(
        &mut self,
        component_idx: usize,
        threshold: T,
        direction: CrossingDirection,
    ) -> SDEProblemSoloutPair<'_, T, Y, D, F, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshold).with_direction(direction);
        SDEProblemSoloutPair::new(self, crossing_solout)
    }

    /// Uses the HyperplaneCrossingSolout method to output points when a specific hyperplane is crossed
    /// Note this does not include the solution of the calculated steps
    ///
    /// # Arguments
    /// * `point` - Point on the hyperplane
    /// * `normal` - Normal vector of the hyperplane
    /// * `extractor` - Function to extract the component from the state vector
    /// * `direction` - Direction of crossing (positive or negative)
    ///
    /// # Returns
    /// * SDE Problem with HyperplaneCrossingSolout function ready for .solve() method
    ///
    pub fn hyperplane_crossing<Y1>(
        &mut self,
        point: Y1,
        normal: Y1,
        extractor: fn(&Y) -> Y1,
        direction: CrossingDirection,
    ) -> SDEProblemSoloutPair<'_, T, Y, D, F, HyperplaneCrossingSolout<T, Y1, Y>>
    where
        Y1: State<T>,
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        SDEProblemSoloutPair::new(self, solout)
    }
}

/// SDEProblemMutRefSoloutPair serves as an intermediate between the SDEProblem struct and a custom solout provided by the user
pub struct SDEProblemMutRefSoloutPair<'a, T, Y, D, F, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
    O: Solout<T, Y, D>,
{
    pub sde_problem: &'a mut SDEProblem<T, Y, D, F>,
    pub solout: &'a mut O,
}

impl<'a, T, Y, D, F, O> SDEProblemMutRefSoloutPair<'a, T, Y, D, F, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
    O: Solout<T, Y, D>,
{
    /// Create a new SDEProblemMutRefSoloutPair
    ///
    /// # Arguments
    /// * `sde_problem` - Reference to the SDE Problem struct
    /// * `solout` - Reference to the solout implementation
    ///
    pub fn new(sde_problem: &'a mut SDEProblem<T, Y, D, F>, solout: &'a mut O) -> Self {
        SDEProblemMutRefSoloutPair {
            sde_problem,
            solout,
        }
    }

    /// Solve the SDE Problem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - StochasticNumericalMethod to use for solving the SDE Problem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: StochasticNumericalMethod<T, Y, D> + Interpolation<T, Y>,
    {
        solve_sde(
            solver,
            &mut self.sde_problem.sde,
            self.sde_problem.t0,
            self.sde_problem.tf,
            &self.sde_problem.y0,
            self.solout,
        )
    }
}

/// SDEProblemSoloutPair serves as an intermediate between the SDEProblem struct and solve_sde when a predefined solout is used
#[derive(Debug)]
pub struct SDEProblemSoloutPair<'a, T, Y, D, F, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
    O: Solout<T, Y, D>,
{
    pub sde_problem: &'a mut SDEProblem<T, Y, D, F>,
    pub solout: O,
}

impl<'a, T, Y, D, F, O> SDEProblemSoloutPair<'a, T, Y, D, F, O>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
    F: SDE<T, Y, D>,
    O: Solout<T, Y, D>,
{
    /// Create a new SDEProblemSoloutPair
    ///
    /// # Arguments
    /// * `sde_problem` - Reference to the SDE Problem struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(sde_problem: &'a mut SDEProblem<T, Y, D, F>, solout: O) -> Self {
        SDEProblemSoloutPair {
            sde_problem,
            solout,
        }
    }

    /// Solve the SDE Problem using the provided solout
    ///
    /// # Arguments
    /// * `solver` - StochasticNumericalMethod to use for solving the SDE Problem
    ///
    /// # Returns
    /// * `Result<Solution<T, Y, D>, Error<T, Y>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if errors or issues are encountered
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, Y, D>, Error<T, Y>>
    where
        S: StochasticNumericalMethod<T, Y, D> + Interpolation<T, Y>,
    {
        solve_sde(
            solver,
            &mut self.sde_problem.sde,
            self.sde_problem.t0,
            self.sde_problem.tf,
            &self.sde_problem.y0,
            &mut self.solout,
        )
    }
}
