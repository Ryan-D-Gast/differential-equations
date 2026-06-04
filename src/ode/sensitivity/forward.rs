use crate::linalg::Matrix;
use crate::ode::ODE;
use crate::ode::sensitivity::traits::ParametrizedODE;
use crate::traits::{Real, State};
use std::cell::RefCell;

/// Forward Sensitivity System Wrapper.
///
/// This wrapper encapsulates a parametrized system of differential equations
/// and automatically constructs the augmented state system for calculating
/// parametric sensitivities.
///
/// # Mathematical Formulation
/// For a system $\frac{dy}{dt} = f(t, y, p)$, the augmented system integrates
/// both the state $y$ and the sensitivity matrix $S = \frac{\partial y}{\partial p}$
/// (row-major flat layout) simultaneously:
///
/// $$\frac{dy}{dt} = f(t, y, p)$$
/// $$\frac{dS}{dt} = J_y(t, y, p) S + J_p(t, y, p)$$
///
/// The augmented state vector layout is `[y_0, ..., y_{n-1}, S_{0,0}, S_{0,1}, ..., S_{n-1,m-1}]`.
///
/// # Lifetimes and Ownership
/// * `F` - Parametrized ODE system type implementing [`ParametrizedODE`]. The wrapper owns
///   the system, which allows it to support both owned systems (e.g. from closures)
///   and borrowed systems (via references since references to parametrized ODEs
///   also implement the trait).
///
/// # Type Parameters
/// * `T` - Scalar type implementing [`Real`].
/// * `Y` - State type representing the system variables.
/// * `P` - State type representing the parameter list.
pub struct ForwardSensitivityOde<F, T: Real, Y: State<T>, P: State<T>> {
    ode: F,
    y_proto: Y,
    j_y: RefCell<Matrix<T>>,
    j_p: RefCell<Matrix<T>>,
    y_cache: RefCell<Y>,
    dydt_cache: RefCell<Y>,
    _marker: std::marker::PhantomData<P>,
}

impl<F, T, Y, P> ForwardSensitivityOde<F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParametrizedODE<T, Y, P>,
{
    /// Creates a new forward sensitivity system wrapper.
    ///
    /// # Arguments
    /// * `ode` - The parametrized ODE system (can be owned or a reference).
    /// * `y_proto` - A prototype instance of state `Y` used for zero-allocation caching.
    ///
    /// # Returns
    /// An instance of `ForwardSensitivityOde`.
    pub fn new(ode: F, y_proto: Y) -> Self {
        let n = y_proto.len();
        let m = ode.parameters().len();
        Self {
            ode,
            y_proto: y_proto.clone(),
            j_y: RefCell::new(Matrix::full(n, n)),
            j_p: RefCell::new(Matrix::full(n, m)),
            y_cache: RefCell::new(y_proto.zeros_like()),
            dydt_cache: RefCell::new(y_proto.zeros_like()),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, T, Y, P, YA> ODE<T, YA> for ForwardSensitivityOde<F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    YA: State<T>,
    F: ParametrizedODE<T, Y, P>,
{
    fn diff(&self, t: T, y_aug: &YA, dydt_aug: &mut YA) {
        let n = self.y_proto.len();
        let m = self.j_p.borrow().dims().1;

        let mut y = self.y_cache.borrow_mut();
        for i in 0..n {
            y.set_component(i, y_aug.get_component(i));
        }

        let mut dydt = self.dydt_cache.borrow_mut();
        self.ode.diff(t, &*y, &mut *dydt);
        for i in 0..n {
            dydt_aug.set_component(i, dydt.get_component(i));
        }

        let mut j_y = self.j_y.borrow_mut();
        self.ode.jacobian(t, &*y, &mut j_y);

        let mut j_p = self.j_p.borrow_mut();
        self.ode.jacobian_p(t, &*y, &mut j_p);

        // dS/dt = J_y * S + J_p
        // S is an n x m matrix stored column-wise or row-wise. Let's do row-wise.
        for r in 0..n {
            for c in 0..m {
                let mut dsdt_rc = j_p[(r, c)];
                for k in 0..n {
                    // S_kc is at index n + k * m + c (row-major S)
                    let s_kc = y_aug.get_component(n + k * m + c);
                    dsdt_rc += j_y[(r, k)] * s_kc;
                }
                dydt_aug.set_component(n + r * m + c, dsdt_rc);
            }
        }
    }
}
