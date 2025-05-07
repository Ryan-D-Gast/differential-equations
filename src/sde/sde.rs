//! Defines system of stochastic differential equations for numerical solvers.
//! The NumericalMethods use this trait to take a input system from the user and solve
//! Includes a stochastic differential equation and optional event function to interupt solver
//! given a condition or event.

use crate::{
    ControlFlag,
    traits::{CallBackData, Real, State},
};

/// SDE Trait for Stochastic Differential Equations
///
/// SDE trait defines the stochastic differential equation dY = a(t,Y)dt + b(t,Y)dW for the solver.
/// The stochastic differential equation is used to solve systems with both deterministic and random components.
/// The trait also includes a solout function to interupt the solver when a condition
/// is met or event occurs.
///
/// # Impl
/// * `drift`     - Deterministic part a(t,Y) of the SDE in form drift(t, &y, &mut dydt).
/// * `diffusion` - Stochastic part b(t,Y) of the SDE in form diffusion(t, &y, &mut dydw).
/// * `noise`     - Generates the random noise increments for the SDE.
/// * `event`     - Solout function to interupt solver when condition is met or event occurs.
///
/// Note that the event function is optional and can be left out when implementing
/// in which case it will use a default implementation.
///
#[allow(unused_variables)]
pub trait SDE<T = f64, V = f64, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Drift function a(t,Y) - deterministic part of the SDE
    ///
    /// A stochastic differential equation (SDE) takes an independent variable
    /// which in this case is 't' as it is typically time and a dependent variable
    /// which is a vector of values 'y'. The drift term represents the deterministic
    /// component of the SDE that describes the expected change in the system over time.
    ///
    /// For efficiency and ergonomics the drift is calculated from an argument
    /// of a mutable reference to the drift vector dydt. This allows for
    /// calculations to be done in place which is more efficient as iterative
    /// SDE solvers require this term to be calculated at each step.
    ///
    /// # Arguments
    /// * `t`    - Independent variable point.
    /// * `y`    - Dependent variable point.
    /// * `dydt` - Drift term output.
    ///
    fn drift(&self, t: T, y: &V, dydt: &mut V);

    /// Diffusion function b(t,Y) - stochastic part of the SDE
    ///
    /// This represents the random/stochastic component of the SDE that describes
    /// how noise or random fluctuations affect the system. The diffusion term
    /// is multiplied by a Wiener process (dW) in the SDE formulation.
    ///
    /// # Arguments
    /// * `t`    - Independent variable point.
    /// * `y`    - Dependent variable point.
    /// * `dydw` - Diffusion term output.
    ///
    fn diffusion(&self, t: T, y: &V, dydw: &mut V);

    /// Noise function - generates random noise increments for the SDE
    ///
    /// This function allows custom control over how noise is generated for the SDE.
    /// Users should implement this to generate appropriate random increments for their SDE.
    /// For standard Wiener process, this is typically normal distribution with mean 0
    /// and standard deviation sqrt(dt).
    ///
    /// Users are responsible for initializing and managing their own random number
    /// generators, allowing full control over seeding for reproducible results.
    ///
    /// # Arguments
    /// * `dt` - Time step size.
    /// * `dw` - Vector to store generated noise increments.
    ///
    fn noise(&self, dt: T, dw: &mut V);

    /// Event function to detect significant conditions during integration.
    ///
    /// Called after each step to detect events like threshold crossings,
    /// singularities, or other mathematically/physically significant conditions.
    /// Can be used to terminate integration when conditions occur.
    ///
    /// # Arguments
    /// * `t`    - Current independent variable point.
    /// * `y`    - Current dependent variable point.
    ///
    /// # Returns
    /// * `ControlFlag` - Command to continue or stop solver.
    ///
    fn event(&self, t: T, y: &V) -> ControlFlag<D> {
        ControlFlag::Continue
    }
}
