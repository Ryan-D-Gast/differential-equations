//! Defines partial differential equations for method-of-lines discretization.

use crate::traits::{DefaultState, Real, State};

/// Multi-dimensional PDE in conservative form.
///
/// The method-of-lines adapter discretizes equations of the form:
///
/// ```text
/// u_t = div(flux(t, x, u, grad_u)) + source(t, x, u)
/// ```
///
/// `U` is the local field value at one spatial node. `D` is the spatial
/// dimension. Scalar PDEs use `T`; vector-field PDEs can use arrays, vectors,
/// or another [`State`] backend.
pub trait PDE<T = f64, U = DefaultState<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
{
    /// Evaluate the spatial flux at a point.
    ///
    /// `grad_u[axis]` is the local approximation of the field derivative along
    /// that spatial axis. `flux[axis]` is the flux contribution along that axis.
    fn flux(&self, t: T, x: &[T; D], u: &U, grad_u: &[U; D], flux: &mut [U; D]);

    /// Evaluate the source term.
    fn source(&self, _t: T, _x: &[T; D], _u: &U, source: &mut U) {
        source.fill(T::zero());
    }
}

impl<EqType, T, U, const D: usize> PDE<T, U, D> for &EqType
where
    T: Real,
    U: State<T>,
    EqType: PDE<T, U, D> + ?Sized,
{
    fn flux(&self, t: T, x: &[T; D], u: &U, grad_u: &[U; D], flux: &mut [U; D]) {
        (*self).flux(t, x, u, grad_u, flux);
    }

    fn source(&self, t: T, x: &[T; D], u: &U, source: &mut U) {
        (*self).source(t, x, u, source);
    }
}
