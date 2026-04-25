//! Adjoint sensitivity system
//!
//! Provides the system for evaluating gradients backwards in time.

use crate::{
    adjoint::cost::CostFunction,
    linalg::Matrix,
    ode::ODE,
    solution::Solution,
    traits::{Real, State},
};

/// Trait for parameterized ODEs, needed for Adjoint Sensitivity Analysis.
pub trait ParameterizedODE<T: Real, Y: State<T>, P: State<T>>: ODE<T, Y> {
    /// Differential Equation with respect to parameters: dy/dt = f(t, y, p).
    /// Defaults to ignoring `p` and just calling `ODE::diff`.
    /// Implementers should override this to use parameters.
    fn diff_p(&self, t: T, y: &Y, p: &P, dydt: &mut Y) {
        let _ = p; // ignore default
        self.diff(t, y, dydt);
    }

    /// Jacobian with respect to parameters J_p = df/dp
    /// Pre-sized to `dim(y) x dim(p)`.
    fn jacobian_p(&self, t: T, y: &Y, p: &P, jp: &mut Matrix<T>) {
        let dim_y = y.len();
        let dim_p = p.len();
        let mut p_perturbed = *p;
        let mut f_perturbed = Y::zeros();
        let mut f_origin = Y::zeros();

        self.diff_p(t, y, p, &mut f_origin);

        let eps = T::default_epsilon().sqrt();

        for j_col in 0..dim_p {
            let p_original_j = p.get(j_col);
            let perturbation = eps * p_original_j.abs().max(T::one());

            p_perturbed.set(j_col, p_original_j + perturbation);
            self.diff_p(t, y, &p_perturbed, &mut f_perturbed);
            p_perturbed.set(j_col, p_original_j);

            for i_row in 0..dim_y {
                jp[(i_row, j_col)] = (f_perturbed.get(i_row) - f_origin.get(i_row)) / perturbation;
            }
        }
    }

    /// Jacobian with respect to state J_y = df/dy
    /// Same as `ODE::jacobian` but may take parameters into account if parameterized.
    fn jacobian_y(&self, t: T, y: &Y, p: &P, jy: &mut Matrix<T>) {
        let dim_y = y.len();
        let mut y_perturbed = *y;
        let mut f_perturbed = Y::zeros();
        let mut f_origin = Y::zeros();

        self.diff_p(t, y, p, &mut f_origin);

        let eps = T::default_epsilon().sqrt();

        for j_col in 0..dim_y {
            let y_original_j = y.get(j_col);
            let perturbation = eps * y_original_j.abs().max(T::one());

            y_perturbed.set(j_col, y_original_j + perturbation);
            self.diff_p(t, &y_perturbed, p, &mut f_perturbed);
            y_perturbed.set(j_col, y_original_j);

            for i_row in 0..dim_y {
                jy[(i_row, j_col)] = (f_perturbed.get(i_row) - f_origin.get(i_row)) / perturbation;
            }
        }
    }
}

/// We need to define a combined state vector for lambda and mu.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjointState<T: Real, Y: State<T>, P: State<T>> {
    pub lambda: Y,
    pub mu: P,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Real, Y: State<T>, P: State<T>> AdjointState<T, Y, P> {
    pub fn new(lambda: Y, mu: P) -> Self {
        Self {
            lambda,
            mu,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Add for AdjointState<T, Y, P> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        AdjointState::new(self.lambda + rhs.lambda, self.mu + rhs.mu)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Sub for AdjointState<T, Y, P> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        AdjointState::new(self.lambda - rhs.lambda, self.mu - rhs.mu)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::AddAssign for AdjointState<T, Y, P> {
    fn add_assign(&mut self, rhs: Self) {
        self.lambda += rhs.lambda;
        self.mu += rhs.mu;
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Mul<T> for AdjointState<T, Y, P> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        AdjointState::new(self.lambda * rhs, self.mu * rhs)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Div<T> for AdjointState<T, Y, P> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        AdjointState::new(self.lambda / rhs, self.mu / rhs)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Neg for AdjointState<T, Y, P> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AdjointState::new(-self.lambda, -self.mu)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> State<T> for AdjointState<T, Y, P> {
    fn len(&self) -> usize {
        self.lambda.len() + self.mu.len()
    }

    fn get(&self, i: usize) -> T {
        let n_lambda = self.lambda.len();
        if i < n_lambda {
            self.lambda.get(i)
        } else {
            self.mu.get(i - n_lambda)
        }
    }

    fn set(&mut self, i: usize, value: T) {
        let n_lambda = self.lambda.len();
        if i < n_lambda {
            self.lambda.set(i, value);
        } else {
            self.mu.set(i - n_lambda, value);
        }
    }

    fn zeros() -> Self {
        AdjointState::new(Y::zeros(), P::zeros())
    }
}

/// Adjoint ODE system for evaluating parameter gradients via backward integration.
pub struct AdjointODE<
    'a,
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParameterizedODE<T, Y, P>,
    G: CostFunction<T, Y, P>,
> {
    pub ode: &'a F,
    pub cost: &'a G,
    pub p: &'a P,
    pub forward_solution: &'a Solution<T, Y>,
}

impl<'a, T, Y, P, F, G> AdjointODE<'a, T, Y, P, F, G>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParameterizedODE<T, Y, P>,
    G: CostFunction<T, Y, P>,
{
    fn eval_forward_state(&self, t: T) -> Y {
        let times = &self.forward_solution.t;
        let states = &self.forward_solution.y;

        if times.is_empty() {
            return Y::zeros();
        }

        if t <= times[0] {
            return states[0];
        }

        let n = times.len();
        if t >= times[n - 1] {
            return states[n - 1];
        }

        let mut left = 0;
        let mut right = n - 1;

        while left <= right {
            let mid = left + (right - left) / 2;
            if times[mid] == t {
                return states[mid];
            }
            if times[mid] < t {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        let i = right;
        let t0 = times[i];
        let t1 = times[i + 1];
        let y0 = states[i];
        let y1 = states[i + 1];

        // Re-evaluate derivatives at endpoints to allow for cubic hermite interpolation
        let mut k0 = Y::zeros();
        let mut k1 = Y::zeros();
        self.ode.diff_p(t0, &y0, self.p, &mut k0);
        self.ode.diff_p(t1, &y1, self.p, &mut k1);

        crate::interpolate::cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t)
    }

    fn grad_y_cost(&self, t: T, y: &Y, p: &P) -> Y {
        let mut grad = Y::zeros();
        let eps = T::default_epsilon().sqrt();
        let mut y_perturbed = *y;

        let g0 = self.cost.integrand(t, y, p);

        for i in 0..y.len() {
            let y_orig = y.get(i);
            let perturbation = eps * y_orig.abs().max(T::one());
            y_perturbed.set(i, y_orig + perturbation);
            let g1 = self.cost.integrand(t, &y_perturbed, p);
            grad.set(i, (g1 - g0) / perturbation);
            y_perturbed.set(i, y_orig);
        }

        grad
    }

    fn grad_p_cost(&self, t: T, y: &Y, p: &P) -> P {
        let mut grad = P::zeros();
        let eps = T::default_epsilon().sqrt();
        let mut p_perturbed = *p;

        let g0 = self.cost.integrand(t, y, p);

        for i in 0..p.len() {
            let p_orig = p.get(i);
            let perturbation = eps * p_orig.abs().max(T::one());
            p_perturbed.set(i, p_orig + perturbation);
            let g1 = self.cost.integrand(t, y, &p_perturbed);
            grad.set(i, (g1 - g0) / perturbation);
            p_perturbed.set(i, p_orig);
        }

        grad
    }
}

impl<'a, T, Y, P, F, G> ODE<T, AdjointState<T, Y, P>> for AdjointODE<'a, T, Y, P, F, G>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParameterizedODE<T, Y, P>,
    G: CostFunction<T, Y, P>,
{
    fn diff(&self, t: T, state: &AdjointState<T, Y, P>, dydt: &mut AdjointState<T, Y, P>) {
        let y_t = self.eval_forward_state(t);
        let lambda = &state.lambda;

        // Compute Jacobians
        let dim_y = y_t.len();
        let dim_p = self.p.len();

        let mut j_y = Matrix::zeros(dim_y, dim_y);
        self.ode.jacobian_y(t, &y_t, self.p, &mut j_y);

        let mut j_p = Matrix::zeros(dim_y, dim_p);
        self.ode.jacobian_p(t, &y_t, self.p, &mut j_p);

        // Compute gradients of cost
        let grad_y = self.grad_y_cost(t, &y_t, self.p);
        let grad_p = self.grad_p_cost(t, &y_t, self.p);

        // dλ/dt = -J_y(t)^T λ - ∇_y g(t)
        // dμ/dt = -J_p(t)^T λ + ∇_p g(t)

        // 1. Compute J_y(t)^T * λ
        let mut j_y_t_lambda = Y::zeros();
        for j in 0..dim_y {
            // row of J_y^T is col of J_y
            let mut sum = T::zero();
            for i in 0..dim_y {
                // col of J_y^T is row of J_y
                sum += j_y[(i, j)] * lambda.get(i);
            }
            j_y_t_lambda.set(j, sum);
        }

        // 2. Compute J_p(t)^T * λ
        let mut j_p_t_lambda = P::zeros();
        for j in 0..dim_p {
            // row of J_p^T is col of J_p
            let mut sum = T::zero();
            for i in 0..dim_y {
                sum += j_p[(i, j)] * lambda.get(i);
            }
            j_p_t_lambda.set(j, sum);
        }

        // Assemble final derivatives
        dydt.lambda = -j_y_t_lambda - grad_y;
        dydt.mu = -j_p_t_lambda - grad_p;
    }
}
