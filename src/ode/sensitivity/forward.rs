use crate::linalg::Matrix;
use crate::ode::ODE;
use crate::ode::sensitivity::traits::ParametrizedODE;
use crate::traits::{Real, State};
use std::cell::RefCell;

/// Forward Sensitivity System Wrapper
pub struct ForwardSensitivityOde<'a, F, T: Real, Y: State<T>, P: State<T>> {
    ode: &'a F,
    y_proto: Y,
    j_y: RefCell<Matrix<T>>,
    j_p: RefCell<Matrix<T>>,
    y_cache: RefCell<Y>,
    dydt_cache: RefCell<Y>,
    _marker: std::marker::PhantomData<P>,
}

impl<'a, F, T, Y, P> ForwardSensitivityOde<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParametrizedODE<T, Y, P>,
{
    pub fn new(ode: &'a F, y_proto: Y) -> Self {
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

impl<'a, F, T, Y, P, YA> ODE<T, YA> for ForwardSensitivityOde<'a, F, T, Y, P>
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
