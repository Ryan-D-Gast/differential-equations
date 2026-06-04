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
