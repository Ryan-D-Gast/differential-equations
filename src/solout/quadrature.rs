//! Quadrature solout for computing numerical integrals alongside the ODE.

use super::*;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

/// Trait for defining a quadrature output function to integrate alongside the ODE.
pub trait Quadrature<T: Real, Y: State<T>> {
    /// Quadrature state type
    type Q: State<T>;

    /// The integrand function for the quadrature
    ///
    /// # Arguments
    /// * `t`    - Independent variable point.
    /// * `y`    - Dependent variable point.
    /// * `dqdt` - Derivative point for the quadrature state.
    fn integrand(&self, t: T, y: &Y, dqdt: &mut Self::Q);
}

/// A solout wrapper that computes a numerical quadrature alongside the ODE.
///
/// It uses Simpson's 3/8 rule on the interpolated steps to compute the integral.
pub struct QuadratureSolout<'a, T: Real, Y: State<T>, O, E>
where
    O: Solout<T, Y>,
    E: Quadrature<T, Y>,
{
    base: O,
    quadrature: &'a E,
    q_state: E::Q,
    q_out: Rc<RefCell<Vec<E::Q>>>,
    _marker: PhantomData<Y>,
}

impl<'a, T: Real, Y: State<T>, O, E> QuadratureSolout<'a, T, Y, O, E>
where
    O: Solout<T, Y>,
    E: Quadrature<T, Y>,
{
    /// Creates a new QuadratureSolout.
    pub fn new(base: O, quadrature: &'a E, q0: E::Q, q_out: Rc<RefCell<Vec<E::Q>>>) -> Self {
        // Push initial state
        q_out.borrow_mut().push(q0.clone());

        Self {
            base,
            quadrature,
            q_state: q0.clone(),
            q_out,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Real, Y: State<T>, O, E> Solout<T, Y> for QuadratureSolout<'a, T, Y, O, E>
where
    O: Solout<T, Y>,
    E: Quadrature<T, Y>,
{
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Y,
        y_prev: &Y,
        interpolator: &mut I,
        solution: &mut Solution<T, Y>,
    ) -> ControlFlag<T, Y>
    where
        I: Interpolation<T, Y>,
    {
        // Compute quadrature using Simpson's 3/8 rule over [t_prev, t_curr]
        if t_curr != t_prev {
            let h = t_curr - t_prev;
            let three = T::from_f64(3.0).unwrap();
            let eight = T::from_f64(8.0).unwrap();

            let t1 = t_prev;
            let t2 = t_prev + h / three;
            let t3 = t_prev + T::from_f64(2.0).unwrap() * h / three;
            let t4 = t_curr;

            let y1 = *y_prev;
            let y2 = interpolator.interpolate(t2).unwrap();
            let y3 = interpolator.interpolate(t3).unwrap();
            let y4 = *y_curr;

            let mut f1 = E::Q::zeros();
            let mut f2 = E::Q::zeros();
            let mut f3 = E::Q::zeros();
            let mut f4 = E::Q::zeros();

            self.quadrature.integrand(t1, &y1, &mut f1);
            self.quadrature.integrand(t2, &y2, &mut f2);
            self.quadrature.integrand(t3, &y3, &mut f3);
            self.quadrature.integrand(t4, &y4, &mut f4);

            let mut dq = f1;
            dq += f2 * three;
            dq += f3 * three;
            dq += f4;
            dq = dq * (h / eight);

            self.q_state += dq;
            self.q_out.borrow_mut().push(self.q_state);
        }

        // Delegate to base solout
        self.base
            .solout(t_curr, t_prev, y_curr, y_prev, interpolator, solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivp::Ivp;
    use crate::methods::ExplicitRungeKutta;
    use crate::ode::ODE;
    use nalgebra::{vector, SVector};

    struct TestODE;

    impl ODE<f64, SVector<f64, 1>> for TestODE {
        fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
            dydt[0] = y[0]; // dy/dt = y => y(t) = exp(t)
        }
    }

    impl Quadrature<f64, SVector<f64, 1>> for TestODE {
        type Q = SVector<f64, 1>;

        fn integrand(&self, _t: f64, y: &SVector<f64, 1>, dqdt: &mut Self::Q) {
            // Integrate y(t) with respect to t: Q(t) = \int exp(t) dt = exp(t) - 1
            dqdt[0] = y[0];
        }
    }

    #[test]
    fn test_quadrature_solout() {
        let ode = TestODE;
        let t0 = 0.0;
        let tf = 1.0;
        let y0 = vector![1.0];
        let q0 = vector![0.0];
        let q_out = Rc::new(RefCell::new(Vec::new()));

        let _solution = Ivp::ode(&ode, t0, tf, y0)
            .quadrature(&ode, q0, q_out.clone())
            .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
            .solve()
            .unwrap();

        let q_results = q_out.borrow();
        assert!(q_results.len() > 1);

        // Q(1) should be \int_0^1 exp(t) dt = exp(1) - exp(0) = e - 1
        let expected_q_end = std::f64::consts::E - 1.0;
        let q_end = q_results.last().unwrap()[0];

        // Should be extremely accurate because DOP853 Hermite interpolant is high order
        assert!(
            (q_end - expected_q_end).abs() < 1e-4,
            "Expected {}, got {}",
            expected_q_end,
            q_end
        );
    }
}
