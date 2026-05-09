use crate::interpolate::cubic_hermite_interpolate;
use crate::linalg::Matrix;
use crate::ode::ODE;
use crate::ode::sensitivity::traits::ParametrizedODE;
use crate::solution::Solution;
use crate::traits::{Real, State};
use std::cell::RefCell;

/// Adjoint Sensitivity System Wrapper
pub struct AdjointOde<'a, F, T: Real, Y: State<T>, P: State<T>> {
    ode: &'a F,
    forward_solution: Solution<T, Y>,
    y_proto: Y,
    j_y_t: RefCell<Matrix<T>>,
    j_p_t: RefCell<Matrix<T>>,
    j_y: RefCell<Matrix<T>>,
    j_p: RefCell<Matrix<T>>,
    _marker: std::marker::PhantomData<P>,
}

impl<'a, F, T, Y, P> AdjointOde<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParametrizedODE<T, Y, P>,
{
    pub fn new(ode: &'a F, forward_solution: Solution<T, Y>, y_proto: Y) -> Self {
        let n = y_proto.len();
        let m = ode.parameters().len();
        Self {
            ode,
            forward_solution,
            y_proto,
            j_y_t: RefCell::new(Matrix::full(n, n)),
            j_p_t: RefCell::new(Matrix::full(m, n)),
            j_y: RefCell::new(Matrix::full(n, n)),
            j_p: RefCell::new(Matrix::full(n, m)),
            _marker: std::marker::PhantomData,
        }
    }

    fn interpolate_forward(&self, t: T) -> Y {
        let times = &self.forward_solution.t;
        let states = &self.forward_solution.y;

        if times.is_empty() {
            return self.y_proto.zeros_like();
        }

        if t <= times[0] {
            return states[0].clone();
        }

        if t >= *times.last().unwrap() {
            return states.last().unwrap().clone();
        }

        let upper = times.partition_point(|ti| *ti < t);
        let lower = upper - 1;

        let t0 = times[lower];
        let t1 = times[upper];
        let y0 = &states[lower];
        let y1 = &states[upper];

        let mut k0 = self.y_proto.zeros_like();
        let mut k1 = self.y_proto.zeros_like();
        self.ode.diff(t0, y0, &mut k0);
        self.ode.diff(t1, y1, &mut k1);

        cubic_hermite_interpolate(t0, t1, y0, y1, &k0, &k1, t)
    }
}

impl<'a, F, T, Y, P, YA> ODE<T, YA> for AdjointOde<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    YA: State<T>,
    F: ParametrizedODE<T, Y, P>,
{
    fn diff(&self, t: T, adjoint_state: &YA, dydt_aug: &mut YA) {
        let n = self.y_proto.len();
        let m = self.ode.parameters().len();
        assert_eq!(
            adjoint_state.len(),
            n + m,
            "Adjoint state must have length n + m"
        );

        let mut lambda = self.y_proto.zeros_like();
        for i in 0..n {
            lambda.set_component(i, adjoint_state.get_component(i));
        }

        let y = self.interpolate_forward(t);

        let mut j_y = self.j_y.borrow_mut();
        self.ode.jacobian(t, &y, &mut j_y);

        let mut j_p = self.j_p.borrow_mut();
        self.ode.jacobian_p(t, &y, &mut j_p);

        let mut j_y_t = self.j_y_t.borrow_mut();
        let mut j_p_t = self.j_p_t.borrow_mut();

        for r in 0..n {
            for c in 0..n {
                j_y_t[(c, r)] = j_y[(r, c)];
            }
        }

        for r in 0..n {
            for c in 0..m {
                j_p_t[(c, r)] = j_p[(r, c)];
            }
        }

        for r in 0..n {
            let mut sum = T::zero();
            for c in 0..n {
                sum += j_y_t[(r, c)] * lambda.get_component(c);
            }
            dydt_aug.set_component(r, -sum);
        }

        for r in 0..m {
            let mut sum = T::zero();
            for c in 0..n {
                sum += j_p_t[(r, c)] * lambda.get_component(c);
            }
            dydt_aug.set_component(n + r, -sum);
        }
    }
}
