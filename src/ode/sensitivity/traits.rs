use crate::linalg::Matrix;
use crate::ode::ODE;
use crate::traits::{Real, State};

/// Optional trait to define parametric sensitivities for an ODE.
///
/// Implement this trait to provide analytical Jacobians with respect to system parameters
/// for use in forward or adjoint sensitivity analysis.
///
/// # Type Parameters
/// * `T` - Scalar type implementing [`Real`].
/// * `Y` - State type representing the system variables.
/// * `P` - State type representing the parameters of the system.
///
/// # Examples
/// ```rust,ignore
/// use differential_equations::ode::sensitivity::traits::ParametrizedODE;
/// use differential_equations::prelude::*;
/// use nalgebra::SVector;
///
/// struct Decay { k: f64 }
///
/// impl ODE<f64, SVector<f64, 1>> for Decay {
///     fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
///         dydt[0] = -self.k * y[0];
///     }
/// }
///
/// impl ParametrizedODE<f64, SVector<f64, 1>, SVector<f64, 1>> for Decay {
///     fn parameters(&self) -> SVector<f64, 1> {
///         SVector::from([self.k])
///     }
///
///     fn jacobian_p(&self, _t: f64, y: &SVector<f64, 1>, j: &mut Matrix<f64>) {
///         j[(0, 0)] = -y[0]; // df/dk
///     }
/// }
/// ```
pub trait ParametrizedODE<T: Real, Y: State<T>, P: State<T>>: ODE<T, Y> {
    /// Returns the current parameters of the system.
    ///
    /// # Returns
    /// An instance of the parameter state `P` containing the parameters of the system.
    fn parameters(&self) -> P;

    /// The parameter Jacobian matrix `J_p = df/dp`.
    ///
    /// Users must provide an analytical implementation of this method.
    ///
    /// # Arguments
    /// * `t` - Independent variable (typically time).
    /// * `y` - Current state of the system.
    /// * `j` - Parameter Jacobian matrix. This matrix must be pre-sized by the caller
    ///   to `y.len() x p.len()`.
    fn jacobian_p(&self, t: T, y: &Y, j: &mut Matrix<T>);
}

impl<F, T: Real, Y: State<T>, P: State<T>> ParametrizedODE<T, Y, P> for &F
where
    F: ParametrizedODE<T, Y, P> + ?Sized,
{
    fn parameters(&self) -> P {
        (*self).parameters()
    }

    fn jacobian_p(&self, t: T, y: &Y, j: &mut Matrix<T>) {
        (*self).jacobian_p(t, y, j)
    }
}

/// Internal wrapper for creating a parametrized ODE from closures.
#[derive(Debug, Clone, Copy)]
pub struct ParametrizedOdeFnWrapper<F, JP, P> {
    pub(crate) diff_fn: F,
    pub(crate) jacobian_p_fn: JP,
    pub(crate) parameters: P,
}

impl<F, JP, P> ParametrizedOdeFnWrapper<F, JP, P> {
    /// Creates a new closure-based parametrized ODE.
    pub fn new(diff_fn: F, jacobian_p_fn: JP, parameters: P) -> Self {
        Self {
            diff_fn,
            jacobian_p_fn,
            parameters,
        }
    }
}

impl<T, Y, F, JP, P> ODE<T, Y> for ParametrizedOdeFnWrapper<F, JP, P>
where
    T: Real,
    Y: State<T>,
    F: Fn(T, &Y, &mut Y),
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        (self.diff_fn)(t, y, dydt)
    }
}

impl<T, Y, P, F, JP> ParametrizedODE<T, Y, P> for ParametrizedOdeFnWrapper<F, JP, P>
where
    T: Real,
    Y: State<T>,
    P: State<T> + Clone,
    F: Fn(T, &Y, &mut Y),
    JP: Fn(T, &Y, &mut Matrix<T>),
{
    fn parameters(&self) -> P {
        self.parameters.clone()
    }

    fn jacobian_p(&self, t: T, y: &Y, j: &mut Matrix<T>) {
        (self.jacobian_p_fn)(t, y, j)
    }
}
