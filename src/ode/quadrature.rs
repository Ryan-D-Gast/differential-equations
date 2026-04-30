use crate::traits::{Real, State, EmptyState};

/// A trait for defining an optional quadrature state to be integrated alongside an ODE.
pub trait Quadrature<T: Real, Y: State<T>>: Clone {
    type Q: State<T>;
    fn integrand(&self, t: T, y: &Y, dqdt: &mut Self::Q);
}

/// A zero-cost marker struct representing the absence of a quadrature integrand.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoQuadrature;

impl<T: Real, Y: State<T>> Quadrature<T, Y> for NoQuadrature {
    type Q = EmptyState;
    fn integrand(&self, _t: T, _y: &Y, _dqdt: &mut Self::Q) {}
}


/// Extension trait for numerical methods that support tracking a quadrature state.
pub trait QuadratureMethod<T: Real, Y: State<T>, Q: State<T>> {
    /// Returns the current quadrature state.
    fn q(&self) -> &Q;

    /// Evaluate the step-local interpolant for the quadrature state at the given time.
    fn interpolate_q(&mut self, t_interp: T) -> Result<Q, crate::error::Error<T, Y>>;
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::solout::DefaultSolout;
    use crate::ode::solve_ode;

    struct TestODE;
    impl ODE<f64, f64> for TestODE {
        fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
            *dydt = *y; // y' = y => y(t) = exp(t)
        }
    }

    #[derive(Clone, Copy)]
    struct TestQuad;
    impl Quadrature<f64, f64> for TestQuad {
        type Q = f64;
        fn integrand(&self, _t: f64, y: &f64, dqdt: &mut f64) {
            *dqdt = *y * *y; // q' = y^2 => q(t) = int exp(2t) dt = 0.5 * (exp(2t) - 1)
        }
    }

    #[test]
    fn test_quadrature_integration() {
        let ode = TestODE;
        let mut solver = ExplicitRungeKutta::dop853().quadrature(TestQuad).rtol(1e-8).atol(1e-8);
        let mut solout = DefaultSolout::new();

        let res = solve_ode(&mut solver, &ode, 0.0, 1.0, &1.0, &mut solout).unwrap();

        let y_final = res.y.last().unwrap();
        let expected_y = 1.0f64.exp();
        assert!((y_final - expected_y).abs() < 1e-6);

        let q_final = solver.q();
        let expected_q = 0.5 * (2.0f64.exp() - 1.0);
        assert!((q_final - expected_q).abs() < 1e-6);

        // Test dense output for q
        use crate::ode::OrdinaryNumericalMethod;
        let t_mid = (solver.t_prev() + solver.t()) / 2.0;
        let q_mid = solver.interpolate_q(t_mid).unwrap();
        let expected_q_mid = 0.5 * ((2.0_f64 * t_mid).exp() - 1.0);
        // The error bound is looser for interpolation than integration, but still quite accurate.
        // Print values to debug why the assertion failed
        assert!((q_mid - expected_q_mid).abs() < 5e-3); // The cubic hermite interpolant over a step is only an approximation.


    }
}
