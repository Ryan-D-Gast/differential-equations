//! Initial Value Problem Struct and Constructors

use crate::{
    Error, Solution,
    interpolate::Interpolation, ode::{
        ivp::solve_ivp,
        numerical_method::NumericalMethod,
        ODE,
    }, 
    solout::*, 
    traits::{CallBackData, Real, State}, 
};

/// Initial Value Problem Differential Equation NumericalMethod
///
/// An Initial Value Problem (IVP) takes the form:
/// y' = f(t, y), a <= t <= b, y(a) = alpha
///
/// # Overview
///
/// The IVP struct provides a simple interface for solving differential equations:
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
/// let ivp = IVP::new(ode, t0, tf, y0);
/// let solution = ivp.solve(&mut solver).unwrap();
///
/// // Advanced output control:
/// let solution = ivp.even(0.1).solve(&mut solver).unwrap();
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
/// * `new(ode, t0, tf, y0)` - Create a new IVP
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
/// let ivp = IVP::new(ode, 0.0, 10.0, vector![1.0, 0.0]);
/// let results = ivp.solve(&mut method).unwrap();
///
/// // Advanced: evenly spaced output with 0.1 time intervals
/// let results = ivp.dense(4).solve(&mut method).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct IVP<T, V, D, F>
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
    pub y0: V, // Initial State Vector.

    // Phantom Data for Users event output
    _event_output_type: std::marker::PhantomData<D>,
}

impl<T, V, D, F> IVP<T, V, D, F>
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
    /// * IVP Problem ready to be solved.
    ///
    pub fn new(ode: F, t0: T, tf: T, y0: V) -> Self {
        IVP {
            ode,
            t0,
            tf,
            y0,
            _event_output_type: std::marker::PhantomData,
        }
    }

    /// Solve the IVP using a default solout, e.g. outputting solutions at calculated steps.
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Status<T, V, D>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Status)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        let mut default_solout = DefaultSolout::new(); // Default solout implementation
        solve_ivp(
            solver,
            &self.ode,
            self.t0,
            self.tf,
            &self.y0,
            &mut default_solout,
        )
    }

    /// Returns an IVP NumericalMethod with the provided solout function for outputting points.
    ///
    /// # Returns
    /// * IVP NumericalMethod with the provided solout function ready for .solve() method.
    ///
    pub fn solout<'a, O: Solout<T, V, D>>(
        &'a self,
        solout: &'a mut O,
    ) -> IVPMutRefSoloutPair<'a, T, V, D, F, O> {
        IVPMutRefSoloutPair::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `dt` - Interval between each output point.
    ///
    /// # Returns
    /// * IVP NumericalMethod with Even Solout function ready for .solve() method.
    ///
    pub fn even(&self, dt: T) -> IVPSoloutPair<'_, T, V, D, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf); // Even solout implementation
        IVPSoloutPair::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    ///
    /// # Returns
    /// * IVP NumericalMethod with Dense Output function ready for .solve() method.
    ///
    pub fn dense(&self, n: usize) -> IVPSoloutPair<'_, T, V, D, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n); // Dense solout implementation
        IVPSoloutPair::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    ///
    /// # Arguments
    /// * `points` - Custom output points.
    ///
    /// # Returns
    /// * IVP NumericalMethod with Custom Time Evaluation function ready for .solve() method.
    ///
    pub fn t_eval(&self, points: Vec<T>) -> IVPSoloutPair<'_, T, V, D, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf); // Custom time evaluation solout implementation
        IVPSoloutPair::new(self, t_eval_solout)
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
    /// * IVP NumericalMethod with CrossingSolout function ready for .solve() method.
    ///
    pub fn crossing(
        &self,
        component_idx: usize,
        threshhold: T,
        direction: CrossingDirection,
    ) -> IVPSoloutPair<'_, T, V, D, F, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshhold).with_direction(direction); // Crossing solout implementation
        IVPSoloutPair::new(self, crossing_solout)
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
    /// * IVP NumericalMethod with HyperplaneCrossingSolout function ready for .solve() method.
    ///
    pub fn hyperplane_crossing<V1>(
        &self,
        point: V1,
        normal: V1,
        extractor: fn(&V) -> V1,
        direction: CrossingDirection,
    ) -> IVPSoloutPair<'_, T, V, D, F, HyperplaneCrossingSolout<T, V1, V>> 
    where 
        V1: State<T>
    {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);

        IVPSoloutPair::new(self, solout)
    }
}

/// IVPMutRefSoloutPair serves as a intermediate between the IVP struct and a custom solout provided by the user.
pub struct IVPMutRefSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    pub ivp: &'a IVP<T, V, D, F>, // Reference to the IVP struct
    pub solout: &'a mut O,           // Reference to the solout implementation
}

impl<'a, T, V, D, F, O> IVPMutRefSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    /// Create a new IVPMutRefSoloutPair
    ///
    /// # Arguments
    /// * `ivp` - Reference to the IVP struct
    ///
    pub fn new(ivp: &'a IVP<T, V, D, F>, solout: &'a mut O) -> Self {
        IVPMutRefSoloutPair { ivp, solout }
    }

    /// Solve the IVP using the provided solout
    ///
    /// # Arguments
    /// * `solver` - NumericalMethod to use for solving the IVP
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Error<T, V>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(&mut self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        solve_ivp(
            solver,
            &self.ivp.ode,
            self.ivp.t0,
            self.ivp.tf,
            &self.ivp.y0,
            self.solout,
        )
    }
}

/// IVPSoloutPair serves as a intermediate between the IVP struct and solve_ivp when a predefined solout is used.
#[derive(Clone, Debug)]
pub struct IVPSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    pub ivp: &'a IVP<T, V, D, F>, // Reference to the IVP struct
    pub solout: O,                   // Solout implementation
}

impl<'a, T, V, D, F, O> IVPSoloutPair<'a, T, V, D, F, O>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: ODE<T, V, D>,
    O: Solout<T, V, D>,
{
    /// Create a new IVPSoloutPair
    ///
    /// # Arguments
    /// * `ivp` - Reference to the IVP struct
    /// * `solout` - Solout implementation
    ///
    pub fn new(ivp: &'a IVP<T, V, D, F>, solout: O) -> Self {
        IVPSoloutPair { ivp, solout }
    }

    /// Solve the IVP using the provided solout
    ///
    /// # Arguments
    /// * `solver` - NumericalMethod to use for solving the IVP
    ///
    /// # Returns
    /// * `Result<Solution<T, V, D>, Error<T, V>>` - `Ok(Solution)` if successful or interrupted by events, `Err(Error)` if an errors or issues such as stiffness are encountered.
    ///
    pub fn solve<S>(mut self, solver: &mut S) -> Result<Solution<T, V, D>, Error<T, V>>
    where
        S: NumericalMethod<T, V, D> + Interpolation<T, V>,
    {
        solve_ivp(
            solver,
            &self.ivp.ode,
            self.ivp.t0,
            self.ivp.tf,
            &self.ivp.y0,
            &mut self.solout,
        )
    }
}
