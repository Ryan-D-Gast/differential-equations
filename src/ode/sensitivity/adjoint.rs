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
    lambda_cache: RefCell<Y>,
    y_cache: RefCell<Y>,
    k0_cache: RefCell<Y>,
    k1_cache: RefCell<Y>,
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
            y_proto: y_proto.clone(),
            j_y_t: RefCell::new(Matrix::full(n, n)),
            j_p_t: RefCell::new(Matrix::full(m, n)),
            j_y: RefCell::new(Matrix::full(n, n)),
            j_p: RefCell::new(Matrix::full(n, m)),
            lambda_cache: RefCell::new(y_proto.zeros_like()),
            y_cache: RefCell::new(y_proto.zeros_like()),
            k0_cache: RefCell::new(y_proto.zeros_like()),
            k1_cache: RefCell::new(y_proto.zeros_like()),
            _marker: std::marker::PhantomData,
        }
    }

    fn interpolate_forward(&self, t: T, out: &mut Y) {
        let times = &self.forward_solution.t;
        let states = &self.forward_solution.y;

        if times.is_empty() {
            out.fill(T::zero());
            return;
        }

        if t <= times[0] {
            out.clone_from(&states[0]);
            return;
        }

        if t >= *times.last().unwrap() {
            out.clone_from(states.last().unwrap());
            return;
        }

        let upper = times.partition_point(|ti| *ti < t);
        let lower = upper - 1;

        let t0 = times[lower];
        let t1 = times[upper];
        let y0 = &states[lower];
        let y1 = &states[upper];

        let mut k0 = self.k0_cache.borrow_mut();
        let mut k1 = self.k1_cache.borrow_mut();
        self.ode.diff(t0, y0, &mut *k0);
        self.ode.diff(t1, y1, &mut *k1);

        let two = T::from_f64(2.0).unwrap();
        let three = T::from_f64(3.0).unwrap();
        let h = t1 - t0;
        let s = (t - t0) / h;
        let s2 = s * s;
        let s3 = s2 * s;
        let h00 = two * s3 - three * s2 + T::one();
        let h10 = s3 - two * s2 + s;
        let h01 = -two * s3 + three * s2;
        let h11 = s3 - s2;

        out.set_linear_combination(&[(y0, h00), (&*k0, h10 * h), (y1, h01), (&*k1, h11 * h)]);
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

        let mut lambda = self.lambda_cache.borrow_mut();
        for i in 0..n {
            lambda.set_component(i, adjoint_state.get_component(i));
        }

        let mut y = self.y_cache.borrow_mut();
        self.interpolate_forward(t, &mut *y);

        let mut j_y = self.j_y.borrow_mut();
        self.ode.jacobian(t, &*y, &mut j_y);

        let mut j_p = self.j_p.borrow_mut();
        self.ode.jacobian_p(t, &*y, &mut j_p);

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
