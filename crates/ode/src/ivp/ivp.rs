//! Initial Value Problem Struct and Constructors

use crate::{System, Solout, Solution, SolverStatus, Solver, EventData};
use crate::solout::*;
use crate::traits::Real;
use super::solve_ivp;
use nalgebra::SMatrix;

/// Initial Value Problem Differential Equation Solver
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
/// use nalgebra::{SVector, vector};
/// 
/// struct LinearEquation {
///    pub a: f64,
///    pub b: f64,
/// }
/// 
/// impl System<f64, 1, 1> for LinearEquation {
///    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
///        dydt[0] = self.a + self.b * y[0];
///   }
/// }
/// 
/// // Create the system and initial conditions
/// let system = LinearEquation { a: 1.0, b: 2.0 };
/// let t0 = 0.0;
/// let tf = 1.0;
/// let y0 = vector![1.0];
/// let mut solver = DOP853::new().rtol(1e-8).atol(1e-6);
/// 
/// // Basic usage:
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solve(&mut solver).unwrap();
/// 
/// // Advanced output control:
/// let solution = ivp.even(0.1).solve(&mut solver).unwrap();
/// ```
/// 
/// # Fields
/// 
/// * `system` - System implementing the differential equation
/// * `t0` - Initial time
/// * `tf` - Final time
/// * `y0` - Initial state vector
/// 
/// # Basic Usage
/// 
/// * `new(system, t0, tf, y0)` - Create a new IVP
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
/// impl System<f64, 2, 1> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///         dydt[0] = y[1];
///         dydt[1] = -self.k * y[0];
///     }
/// }
/// 
/// let system = HarmonicOscillator { k: 1.0 };
/// let mut solver = DOP853::new().rtol(1e-12).atol(1e-12);
/// 
/// // Basic usage with default output points
/// let ivp = IVP::new(system, 0.0, 10.0, vector![1.0, 0.0]);
/// let results = ivp.solve(&mut solver).unwrap();
/// 
/// // Advanced: evenly spaced output with 0.1 time intervals
/// let results = ivp.dense(4).solve(&mut solver).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct IVP<T, const R: usize, const C: usize, E, F>
where 
    T: Real,
    E: EventData,
    F: System<T, R, C, E>,
{
    // Initial Value Problem Fields
    pub system: F, // System containing the Differential Equation and Optional Terminate Function.
    pub t0: T, // Initial Time.
    pub tf: T, // Final Time.
    pub y0: SMatrix<T, R, C>, // Initial State Vector.

    // Phantom Data for Users event output
    _event_output_type: std::marker::PhantomData<E>,
}

impl<T, const R: usize, const C: usize, E, F> IVP<T, R, C, E, F>
where 
    T: Real,
    E: EventData,
    F: System<T, R, C, E>,
{
    /// Create a new Initial Value Problem
    /// 
    /// # Arguments
    /// * `system`  - System containing the Differential Equation and Optional Terminate Function.
    /// * `t0`      - Initial Time.
    /// * `tf`      - Final Time.
    /// * `y0`      - Initial State Vector.
    /// 
    /// # Returns
    /// * IVP Problem ready to be solved.
    /// 
    pub fn new(system: F, t0: T, tf: T, y0: SMatrix<T, R, C>) -> Self {
        IVP {
            system,
            t0,
            tf,
            y0,
            _event_output_type: std::marker::PhantomData,
        }
    }

    /// Solve the IVP using a default solout, e.g. outputting solutions at calculated steps.
    /// 
    /// # Returns
    /// * `Result<Solution<T, V, E>, SolverStatus<T, V, E>>` - `Ok(Solution)` if successful or interrupted by events, `Err(SolverStatus)` if an errors or issues such as stiffness are encountered.
    /// 
    pub fn solve<S>(&self, solver: &mut S) -> Result<Solution<T, R, C, E, DefaultSolout>, SolverStatus<T, R, C, E>>
    where 
        S: Solver<T, R, C, E>,
    {
        let default_solout = crate::solout::DefaultSolout::new(); // Default solout implementation
        solve_ivp(solver, &self.system, self.t0, self.tf, &self.y0, default_solout)
    }

    /// Returns an IVP Solver with the provided solout function for outputting points.
    /// 
    /// # Returns
    /// * IVP Solver with the provided solout function ready for .solve() method.
    /// 
    pub fn solout<O: Solout<T, R, C, E>>(&self, solout: O) -> IVPSolver<'_, T, R, C, E, F, O> {
        IVPSolver::new(self, solout)
    }

    /// Uses the an Even Solout implementation to output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    /// 
    /// # Arguments
    /// * `dt` - Interval between each output point.
    /// 
    /// # Returns
    /// * IVP Solver with Even Solout function ready for .solve() method.
    /// 
    pub fn even(&self, dt: T) -> IVPSolver<'_, T, R, C, E, F, EvenSolout<T>> {
        let even_solout = EvenSolout::new(dt, self.t0, self.tf); // Even solout implementation
        IVPSolver::new(self, even_solout)
    }

    /// Uses the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    /// 
    /// # Arguments
    /// * `n` - Number of interpolation points between each step.
    /// 
    /// # Returns
    /// * IVP Solver with Dense Output function ready for .solve() method.
    /// 
    pub fn dense(&self, n: usize) -> IVPSolver<'_, T, R, C, E, F, DenseSolout> {
        let dense_solout = DenseSolout::new(n); // Dense solout implementation
        IVPSolver::new(self, dense_solout)
    }

    /// Uses the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    /// 
    /// # Arguments
    /// * `points` - Custom output points.
    /// 
    /// # Returns
    /// * IVP Solver with Custom Time Evaluation function ready for .solve() method.
    /// 
    pub fn t_eval(&self, points: Vec<T>) -> IVPSolver<'_, T, R, C, E, F, TEvalSolout<T>> {
        let t_eval_solout = TEvalSolout::new(points, self.t0, self.tf); // Custom time evaluation solout implementation
        IVPSolver::new(self, t_eval_solout)
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
    /// * IVP Solver with CrossingSolout function ready for .solve() method.
    /// 
    pub fn crossing(&self, component_idx: usize, threshhold: T, direction: CrossingDirection) -> IVPSolver<'_, T, R, C, E, F, CrossingSolout<T>> {
        let crossing_solout = CrossingSolout::new(component_idx, threshhold).with_direction(direction); // Crossing solout implementation
        IVPSolver::new(self, crossing_solout)
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
    /// * IVP Solver with HyperplaneCrossingSolout function ready for .solve() method.
    /// 
    pub fn hyperplane_crossing<const R1: usize, const C1: usize>(&self, 
        point: SMatrix<T, R1, C1>, 
        normal: SMatrix<T, R1, C1>,
        extractor: fn(&SMatrix<T, R, C>) -> SMatrix<T, R1, C1>,
        direction: CrossingDirection
    ) -> IVPSolver<'_, T, R, C, E, F, HyperplaneCrossingSolout<T, R1, C1, R, C>> {
        let solout = HyperplaneCrossingSolout::new(
            point, 
            normal,
            extractor
        ).with_direction(direction);
        
        IVPSolver::new(self, solout)
    }
}


/// IVPSolver serves as a intermediate between the IVP struct and solve_ivp.
#[derive(Clone, Debug)]
pub struct IVPSolver<'a, T, const R: usize, const C: usize, E, F, O>
where 
    T: Real,
    E: EventData,
    F: System<T, R, C, E>,
    O: Solout<T, R, C, E>,
{
    // Initial Value Problem Fields References to original IVP
    system: &'a F, // Reference to System
    t0: &'a T,
    tf: &'a T,
    y0: &'a SMatrix<T, R, C>, // Reference to Initial State Vector

    // Solout Function
    solout: O,

    // Phantom Data for Users event output
    _event_output_type: std::marker::PhantomData<E>,
}

impl<'a, T, const R: usize, const C: usize, E, F, O> IVPSolver<'a, T, R, C, E, F, O>
where 
    T: Real,
    E: EventData,
    F: System<T, R, C, E>,
    O: Solout<T, R, C, E>,
{
    /// Create a new IVPSolver
    /// 
    /// # Arguments
    /// * `ivp` - Reference to Initial Value Problem
    /// * `solout` - Solout function to use for outputting points.
    /// 
    /// # Returns
    /// * IVPSolver ready to solve the IVP.
    /// 
    pub fn new(ivp: &'a IVP<T, R, C, E, F>, solout: O) -> Self {
        IVPSolver {
            system: &ivp.system,
            t0: &ivp.t0,
            tf: &ivp.tf,
            y0: &ivp.y0,
            solout,
            _event_output_type: std::marker::PhantomData,
        }
    }

    /// Solve the IVP using the provided solver.
    /// 
    /// # Arguments
    /// * `solver` - Reference to the solver to use for solving the IVP.
    /// 
    /// # Returns
    /// * `Result<Solution<T, R, C, E>, SolverStatus<T, R, C, E>>` - `Ok(Solution)` if successful or interrupted by events, `Err(SolverStatus)` if an errors or issues such as stiffness are encountered.
    /// 
    pub fn solve<S>(self, solver: &mut S) -> Result<Solution<T, R, C, E, O>, SolverStatus<T, R, C, E>>
    where 
        S: Solver<T, R, C, E>
    {
        solve_ivp(solver, self.system, *self.t0, *self.tf, self.y0, self.solout)
    }
}