use std::cell::RefCell;
use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    ode::{ParametrizedODE, ODE},
    traits::{Real, State},
};

/// Forward Sensitivity Analysis wrapper that creates an augmented ODE problem.
///
/// It solves the base equations dy/dt = f(t, y, p)
/// along with sensitivity equations dS/dt = J_y * S + J_p.
pub struct ForwardSensitivityProblem<'a, F, T, YBase>
where
    T: Real,
    YBase: State<T>,
    F: ParametrizedODE<T, YBase>,
{
    equation: &'a F,
    state_dim: usize,
    num_params: usize,
    jy_cache: RefCell<Matrix<T>>,
    jp_cache: RefCell<Matrix<T>>,
    f_base_cache: RefCell<YBase>,
    y_base_cache: RefCell<YBase>,
    _marker: PhantomData<T>,
}

impl<'a, F, T, YBase> ForwardSensitivityProblem<'a, F, T, YBase>
where
    T: Real,
    YBase: State<T>,
    F: ParametrizedODE<T, YBase>,
{
    /// Create a new FSA problem wrapper for the given parametrized ODE.
    /// `state_dim` must match the length of the base state YBase.
    pub fn new(equation: &'a F, state_dim: usize) -> Self {
        let num_params = equation.num_params();
        Self {
            equation,
            state_dim,
            num_params,
            jy_cache: RefCell::new(Matrix::zeros(state_dim, state_dim)),
            jp_cache: RefCell::new(Matrix::zeros(state_dim, num_params)),
            f_base_cache: RefCell::new(YBase::zeros()),
            y_base_cache: RefCell::new(YBase::zeros()),
            _marker: PhantomData,
        }
    }
}

impl<'a, F, T, YBase, YAug> ODE<T, YAug> for ForwardSensitivityProblem<'a, F, T, YBase>
where
    T: Real,
    YBase: State<T>,
    YAug: State<T>,
    F: ParametrizedODE<T, YBase>,
{
    fn diff(&self, t: T, y_aug: &YAug, dy_aug_dt: &mut YAug) {
        let dim = self.state_dim;
        let p = self.num_params;

        debug_assert_eq!(y_aug.len(), dim + dim * p, "Augmented state has incorrect dimension");

        // 1. Extract base state y
        let mut y_base = self.y_base_cache.borrow_mut();
        for i in 0..dim {
            y_base.set(i, y_aug.get(i));
        }

        // 2. Compute base derivative f(t, y) and Jacobians
        let mut f_base = self.f_base_cache.borrow_mut();
        self.equation.diff(t, &y_base, &mut f_base);

        let mut jy = self.jy_cache.borrow_mut();
        self.equation.jacobian(t, &y_base, &mut jy);

        let mut jp = self.jp_cache.borrow_mut();
        self.equation.jacobian_p(t, &y_base, &mut jp);

        // 3. Set dy/dt in the augmented derivative
        for i in 0..dim {
            dy_aug_dt.set(i, f_base.get(i));
        }

        // 4. Compute dS/dt = Jy * S + Jp
        // S is a (dim x p) matrix, stored row-major or column-major in y_aug.
        // Let's store S in column-major order: parameter sensitivities sequentially.
        // Or row-major? Wait, y_aug is just a vector.
        // We'll use column-major:
        // for each param j:
        //   S_j is a vector of length `dim` starting at `dim + j * dim`
        //   dS_j/dt = Jy * S_j + Jp_j
        for j in 0..p {
            let offset = dim + j * dim;
            for i in 0..dim {
                // Compute (Jy * S_j)_i
                let mut jy_s_i = T::zero();
                for k in 0..dim {
                    let s_kj = y_aug.get(offset + k);
                    jy_s_i += jy[(i, k)] * s_kj;
                }

                // Add (Jp)_ij
                let ds_ij = jy_s_i + jp[(i, j)];
                dy_aug_dt.set(offset + i, ds_ij);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use nalgebra::{SVector, vector};

    // Test system: Exponential growth with parameter
    // dy/dt = k * y
    // Base State: [y], Param: [k]
    // Analytic Sensitivity: S = dy/dk
    // dS/dt = Jy * S + Jp
    // Jy = [k], Jp = [y]
    // Therefore: dS/dt = k * S + y
    // Exact solutions for y0=1, k=2, t=1:
    // y(t) = exp(kt)
    // S(t) = t * exp(kt)

    struct ExponentialParametrized {
        k: f64,
    }

    impl ODE<f64, SVector<f64, 1>> for ExponentialParametrized {
        fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
            dydt[0] = self.k * y[0];
        }
    }

    impl ParametrizedODE<f64, SVector<f64, 1>> for ExponentialParametrized {
        fn num_params(&self) -> usize {
            1
        }

        fn jacobian_p(&self, _t: f64, y: &SVector<f64, 1>, jp: &mut Matrix<f64>) {
            // dy/dk = y
            jp[(0, 0)] = y[0];
        }
    }

    #[test]
    fn test_forward_sensitivity_problem() {
        let system = ExponentialParametrized { k: 2.0 };
        let fsa_problem = ForwardSensitivityProblem::new(&system, 1);

        let t0 = 0.0;
        let tf = 1.0;

        // y0_aug = [y0, S0] = [1.0, 0.0]
        let y0_aug = vector![1.0, 0.0];

        let method = ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12);

        let solution = Ivp::ode(&fsa_problem, t0, tf, y0_aug)
            .method(method)
            .solve()
            .unwrap();

        let y_final = solution.y.last().unwrap();
        let y_val = y_final[0];
        let s_val = y_final[1];

        let y_exact = (2.0f64 * 1.0).exp();
        let s_exact = 1.0 * (2.0f64 * 1.0).exp();

        assert!((y_val - y_exact).abs() < 1e-6);
        assert!((s_val - s_exact).abs() < 1e-6);
    }
}
