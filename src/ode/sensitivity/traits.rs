use crate::linalg::Matrix;
use crate::ode::ODE;
use crate::traits::{Real, State};

/// Optional trait to define parametric sensitivities for an ODE.
///
/// Implement this trait to provide analytical Jacobians with respect to system parameters
/// for use in forward or adjoint sensitivity analysis.
pub trait ParametrizedODE<T: Real, Y: State<T>, P: State<T>>: ODE<T, Y> {
    /// Returns the current parameters of the system.
    fn parameters(&self) -> P;

    /// The parameter Jacobian matrix `J_p = df/dp`.
    ///
    /// The output matrix `j` must be pre-sized to `y.len() x p.len()`.
    /// Users must provide an analytical implementation of this method.
    fn jacobian_p(&self, t: T, y: &Y, j: &mut Matrix<T>);
}
