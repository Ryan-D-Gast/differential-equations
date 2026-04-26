//! Sensitivity analysis for ordinary differential equations.

use std::cell::RefCell;

use crate::{
    error::Error,
    interpolate::Interpolation,
    linalg::Matrix,
    ode::{ODE, OrdinaryNumericalMethod, solve_ode},
    solout::{DefaultSolout, DenseSolout},
    solution::Solution,
    traits::{Real, State},
};

/// ODE extension for systems that depend on an explicit parameter state.
///
/// This trait is shared by forward and adjoint sensitivity analysis so users do
/// not need to implement two different parameter APIs.
pub trait ODEParameters<T: Real, Y: State<T>, P: State<T>>: ODE<T, Y> {
    /// Evaluate `dy/dt = f(t, y, params)`.
    ///
    /// The default ignores `params` and delegates to [`ODE::diff`].
    fn diff_with_params(&self, t: T, y: &Y, params: &P, dydt: &mut Y) {
        let _ = params;
        self.diff(t, y, dydt);
    }

    /// Fill `jy` with the state Jacobian `df/dy`.
    ///
    /// The default uses forward finite differences around [`Self::diff_with_params`].
    fn jacobian_state(&self, t: T, y: &Y, params: &P, jy: &mut Matrix<T>) {
        let dim_y = y.len();
        let mut y_perturbed = y.clone();
        let mut f_perturbed = y.zeros_like();
        let mut f_origin = y.zeros_like();
        self.diff_with_params(t, y, params, &mut f_origin);

        let eps = T::default_epsilon().sqrt();
        for j_col in 0..dim_y {
            let y_original = y.get(j_col);
            let perturbation = eps * y_original.abs().max(T::one());
            y_perturbed.set(j_col, y_original + perturbation);
            self.diff_with_params(t, &y_perturbed, params, &mut f_perturbed);
            y_perturbed.set(j_col, y_original);

            for i_row in 0..dim_y {
                jy[(i_row, j_col)] = (f_perturbed.get(i_row) - f_origin.get(i_row)) / perturbation;
            }
        }
    }

    /// Fill `jp` with the parameter Jacobian `df/dp`.
    ///
    /// The matrix is pre-sized to `y.len() x p.len()`. The default uses forward
    /// finite differences; override this with an analytic Jacobian when possible.
    fn jacobian_params(&self, t: T, y: &Y, params: &P, jp: &mut Matrix<T>) {
        let dim_y = y.len();
        let dim_p = params.len();
        let mut params_perturbed = params.clone();
        let mut f_perturbed = y.zeros_like();
        let mut f_origin = y.zeros_like();
        self.diff_with_params(t, y, params, &mut f_origin);

        let eps = T::default_epsilon().sqrt();
        for j_col in 0..dim_p {
            let param_original = params.get(j_col);
            let perturbation = eps * param_original.abs().max(T::one());
            params_perturbed.set(j_col, param_original + perturbation);
            self.diff_with_params(t, y, &params_perturbed, &mut f_perturbed);
            params_perturbed.set(j_col, param_original);

            for i_row in 0..dim_y {
                jp[(i_row, j_col)] = (f_perturbed.get(i_row) - f_origin.get(i_row)) / perturbation;
            }
        }
    }
}

/// Cost functional used by adjoint sensitivity analysis.
///
/// The total cost is interpreted as a terminal term plus the integral of
/// [`Self::integrand`] over the integration interval.
pub trait AdjointCost<T: Real, Y: State<T>, P: State<T>> {
    /// Continuous integrand `g(t, y, p)`.
    fn integrand(&self, _t: T, _y: &Y, _p: &P) -> T {
        T::zero()
    }

    /// Terminal contribution `h(tf, y(tf), p)`.
    fn terminal(&self, _tf: T, _yf: &Y, _p: &P) -> T {
        T::zero()
    }

    /// Fill `grad_y` with `dg/dy`.
    fn integrand_gradient_y(&self, t: T, y: &Y, p: &P, grad_y: &mut Y) {
        finite_difference_y(|yi| self.integrand(t, yi, p), y, grad_y);
    }

    /// Fill `grad_p` with `dg/dp`.
    fn integrand_gradient_p(&self, t: T, y: &Y, p: &P, grad_p: &mut P) {
        finite_difference_p(|pi| self.integrand(t, y, pi), p, grad_p);
    }

    /// Fill `grad_y` with `dh/dy` at the terminal time.
    fn terminal_gradient_y(&self, tf: T, yf: &Y, p: &P, grad_y: &mut Y) {
        finite_difference_y(|yi| self.terminal(tf, yi, p), yf, grad_y);
    }

    /// Fill `grad_p` with `dh/dp` at the terminal time.
    fn terminal_gradient_p(&self, tf: T, yf: &Y, p: &P, grad_p: &mut P) {
        finite_difference_p(|pi| self.terminal(tf, yf, pi), p, grad_p);
    }
}

/// ODE wrapper for forward sensitivity analysis.
///
/// The augmented state is stored as `[y, S_0, S_1, ...]`, where each `S_j`
/// contains `dy/dp_j` and has length `y.len()`.
pub struct ForwardSensitivityODE<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
{
    equation: &'a F,
    params: &'a P,
    state_dim: usize,
    param_dim: usize,
    y_cache: RefCell<Y>,
    f_cache: RefCell<Y>,
    jy_cache: RefCell<Matrix<T>>,
    jp_cache: RefCell<Matrix<T>>,
}

impl<'a, F, T, Y, P> ForwardSensitivityODE<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
{
    /// Create a forward sensitivity wrapper.
    pub fn new(equation: &'a F, params: &'a P, y_template: Y) -> Self {
        let state_dim = y_template.len();
        let param_dim = params.len();
        let f_cache = y_template.zeros_like();
        Self {
            equation,
            params,
            state_dim,
            param_dim,
            y_cache: RefCell::new(y_template),
            f_cache: RefCell::new(f_cache),
            jy_cache: RefCell::new(Matrix::zeros(state_dim, state_dim)),
            jp_cache: RefCell::new(Matrix::zeros(state_dim, param_dim)),
        }
    }

    /// Length required for the augmented state.
    pub fn augmented_len(&self) -> usize {
        self.state_dim + self.state_dim * self.param_dim
    }
}

impl<'a, F, T, Y, P, YAug> ODE<T, YAug> for ForwardSensitivityODE<'a, F, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    YAug: State<T>,
    F: ODEParameters<T, Y, P>,
{
    fn diff(&self, t: T, y_aug: &YAug, dy_aug_dt: &mut YAug) {
        let dim_y = self.state_dim;
        let dim_p = self.param_dim;
        debug_assert_eq!(y_aug.len(), self.augmented_len());

        let mut y = self.y_cache.borrow_mut();
        for i in 0..dim_y {
            y.set(i, y_aug.get(i));
        }

        let mut f = self.f_cache.borrow_mut();
        self.equation.diff_with_params(t, &y, self.params, &mut f);

        let mut jy = self.jy_cache.borrow_mut();
        self.equation.jacobian_state(t, &y, self.params, &mut jy);

        let mut jp = self.jp_cache.borrow_mut();
        self.equation.jacobian_params(t, &y, self.params, &mut jp);

        for i in 0..dim_y {
            dy_aug_dt.set(i, f.get(i));
        }

        for j in 0..dim_p {
            let offset = dim_y + j * dim_y;
            for i in 0..dim_y {
                let mut sum = T::zero();
                for k in 0..dim_y {
                    sum += jy[(i, k)] * y_aug.get(offset + k);
                }
                dy_aug_dt.set(offset + i, sum + jp[(i, j)]);
            }
        }
    }
}

/// State used for adjoint sensitivity integration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjointState<T: Real, Y: State<T>, P: State<T>> {
    /// State adjoint `lambda`.
    pub lambda: Y,
    /// Parameter gradient accumulator `mu`.
    pub mu: P,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Real, Y: State<T>, P: State<T>> AdjointState<T, Y, P> {
    /// Create an adjoint state from state and parameter components.
    pub fn new(lambda: Y, mu: P) -> Self {
        Self {
            lambda,
            mu,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Real, Y: State<T>, P: State<T>> State<T> for AdjointState<T, Y, P> {
    fn len(&self) -> usize {
        self.lambda.len() + self.mu.len()
    }

    fn get(&self, i: usize) -> T {
        let lambda_len = self.lambda.len();
        if i < lambda_len {
            self.lambda.get(i)
        } else {
            self.mu.get(i - lambda_len)
        }
    }

    fn set(&mut self, i: usize, value: T) {
        let lambda_len = self.lambda.len();
        if i < lambda_len {
            self.lambda.set(i, value);
        } else {
            self.mu.set(i - lambda_len, value);
        }
    }

    fn zeros() -> Self {
        Self::new(Y::zeros(), P::zeros())
    }

    fn zeros_like(&self) -> Self {
        Self::new(self.lambda.zeros_like(), self.mu.zeros_like())
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Add for AdjointState<T, Y, P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.lambda + rhs.lambda, self.mu + rhs.mu)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Sub for AdjointState<T, Y, P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.lambda - rhs.lambda, self.mu - rhs.mu)
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
        Self::new(self.lambda * rhs, self.mu * rhs)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Div<T> for AdjointState<T, Y, P> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.lambda / rhs, self.mu / rhs)
    }
}

impl<T: Real, Y: State<T>, P: State<T>> core::ops::Neg for AdjointState<T, Y, P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.lambda, -self.mu)
    }
}

/// Output from an adjoint sensitivity solve.
#[derive(Debug, Clone)]
pub struct AdjointSolution<T: Real, Y: State<T>, P: State<T>> {
    /// Forward primal solution.
    pub forward: Solution<T, Y>,
    /// Reverse-time adjoint trajectory.
    pub adjoint: Solution<T, AdjointState<T, Y, P>>,
    /// Gradient with respect to the initial state.
    pub grad_y0: Y,
    /// Gradient with respect to parameters.
    pub grad_p: P,
}

struct AdjointProblem<'a, F, C, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
    C: AdjointCost<T, Y, P>,
{
    equation: &'a F,
    cost: &'a C,
    params: &'a P,
    forward: &'a Solution<T, Y>,
}

impl<'a, F, C, T, Y, P> ODE<T, AdjointState<T, Y, P>> for AdjointProblem<'a, F, C, T, Y, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
    C: AdjointCost<T, Y, P>,
{
    fn diff(&self, t: T, state: &AdjointState<T, Y, P>, dydt: &mut AdjointState<T, Y, P>) {
        let y = interpolate_solution(self.forward, self.equation, self.params, t);
        let dim_y = y.len();
        let dim_p = self.params.len();

        let mut jy = Matrix::zeros(dim_y, dim_y);
        self.equation.jacobian_state(t, &y, self.params, &mut jy);

        let mut jp = Matrix::zeros(dim_y, dim_p);
        self.equation.jacobian_params(t, &y, self.params, &mut jp);

        let mut grad_y = y.zeros_like();
        self.cost
            .integrand_gradient_y(t, &y, self.params, &mut grad_y);

        let mut grad_p = self.params.zeros_like();
        self.cost
            .integrand_gradient_p(t, &y, self.params, &mut grad_p);

        for j in 0..dim_y {
            let mut sum = T::zero();
            for i in 0..dim_y {
                sum += jy[(i, j)] * state.lambda.get(i);
            }
            dydt.lambda.set(j, -sum - grad_y.get(j));
        }

        for j in 0..dim_p {
            let mut sum = T::zero();
            for i in 0..dim_y {
                sum += jp[(i, j)] * state.lambda.get(i);
            }
            dydt.mu.set(j, -sum + grad_p.get(j));
        }
    }
}

/// Solve an adjoint sensitivity problem using the same parameter API as FSA.
#[allow(clippy::type_complexity)]
pub fn solve_adjoint_sensitivity<T, Y, P, F, C, ForwardMethod, BackwardMethod>(
    forward_method: &mut ForwardMethod,
    backward_method: &mut BackwardMethod,
    equation: &F,
    cost: &C,
    t0: T,
    tf: T,
    y0: &Y,
    params: &P,
) -> Result<AdjointSolution<T, Y, P>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
    C: AdjointCost<T, Y, P>,
    ForwardMethod: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    BackwardMethod:
        OrdinaryNumericalMethod<T, AdjointState<T, Y, P>> + Interpolation<T, AdjointState<T, Y, P>>,
{
    let mut forward_solout = DenseSolout::new(8);
    let forward = solve_ode(
        forward_method,
        &ODEParametersAdapter { equation, params },
        t0,
        tf,
        y0,
        &mut forward_solout,
    )?;

    let yf = match forward.y.last() {
        Some(yf) => yf,
        None => {
            return Err(Error::BadInput {
                msg: "Forward solve did not produce a terminal state.".to_string(),
            });
        }
    };

    let mut lambda_tf = yf.zeros_like();
    cost.terminal_gradient_y(tf, yf, params, &mut lambda_tf);

    let mut mu_tf = params.zeros_like();
    cost.terminal_gradient_p(tf, yf, params, &mut mu_tf);

    let adjoint_y0 = AdjointState::new(lambda_tf, mu_tf);
    let adjoint_problem = AdjointProblem {
        equation,
        cost,
        params,
        forward: &forward,
    };
    let mut backward_solout = DefaultSolout::new();
    let adjoint = solve_ode(
        backward_method,
        &adjoint_problem,
        tf,
        t0,
        &adjoint_y0,
        &mut backward_solout,
    )
    .map_err(adjoint_error_to_state_error)?;

    let terminal = match adjoint.y.last() {
        Some(state) => state.clone(),
        None => {
            return Err(Error::BadInput {
                msg: "Adjoint solve did not produce an initial-time state.".to_string(),
            });
        }
    };

    Ok(AdjointSolution {
        forward,
        adjoint,
        grad_y0: terminal.lambda,
        grad_p: terminal.mu,
    })
}

struct ODEParametersAdapter<'a, F, P> {
    equation: &'a F,
    params: &'a P,
}

impl<'a, F, T, Y, P> ODE<T, Y> for ODEParametersAdapter<'a, F, P>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        self.equation.diff_with_params(t, y, self.params, dydt);
    }

    fn jacobian(&self, t: T, y: &Y, j: &mut Matrix<T>) {
        self.equation.jacobian_state(t, y, self.params, j);
    }
}

fn finite_difference_y<T, Y>(mut scalar: impl FnMut(&Y) -> T, y: &Y, grad_y: &mut Y)
where
    T: Real,
    Y: State<T>,
{
    let g0 = scalar(y);
    let eps = T::default_epsilon().sqrt();
    let mut y_perturbed = y.clone();
    for i in 0..y.len() {
        let y_original = y.get(i);
        let perturbation = eps * y_original.abs().max(T::one());
        y_perturbed.set(i, y_original + perturbation);
        let g1 = scalar(&y_perturbed);
        grad_y.set(i, (g1 - g0) / perturbation);
        y_perturbed.set(i, y_original);
    }
}

fn finite_difference_p<T, P>(mut scalar: impl FnMut(&P) -> T, p: &P, grad_p: &mut P)
where
    T: Real,
    P: State<T>,
{
    let g0 = scalar(p);
    let eps = T::default_epsilon().sqrt();
    let mut p_perturbed = p.clone();
    for i in 0..p.len() {
        let p_original = p.get(i);
        let perturbation = eps * p_original.abs().max(T::one());
        p_perturbed.set(i, p_original + perturbation);
        let g1 = scalar(&p_perturbed);
        grad_p.set(i, (g1 - g0) / perturbation);
        p_perturbed.set(i, p_original);
    }
}

fn interpolate_solution<T, Y, P, F>(solution: &Solution<T, Y>, equation: &F, params: &P, t: T) -> Y
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ODEParameters<T, Y, P>,
{
    let first_t = solution.t[0];
    let last_idx = solution.t.len() - 1;
    let last_t = solution.t[last_idx];

    if t <= first_t {
        return solution.y[0].clone();
    }
    if t >= last_t {
        return solution.y[last_idx].clone();
    }

    let upper = solution.t.partition_point(|ti| *ti < t);
    if solution.t[upper] == t {
        return solution.y[upper].clone();
    }

    let lower = upper - 1;
    let t0 = solution.t[lower];
    let t1 = solution.t[upper];
    let y0 = solution.y[lower].clone();
    let y1 = solution.y[upper].clone();
    let mut k0 = y0.zeros_like();
    let mut k1 = y1.zeros_like();
    equation.diff_with_params(t0, &y0, params, &mut k0);
    equation.diff_with_params(t1, &y1, params, &mut k1);
    crate::interpolate::cubic_hermite_interpolate(t0, t1, &y0, &y1, &k0, &k1, t)
}

fn adjoint_error_to_state_error<T, Y, P>(err: Error<T, AdjointState<T, Y, P>>) -> Error<T, Y>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
{
    match err {
        Error::BadInput { msg } => Error::BadInput { msg },
        Error::MaxSteps { t, y } => Error::MaxSteps { t, y: y.lambda },
        Error::StepSize { t, y } => Error::StepSize { t, y: y.lambda },
        Error::Stiffness { t, y } => Error::Stiffness { t, y: y.lambda },
        Error::OutOfBounds {
            t_interp,
            t_prev,
            t_curr,
        } => Error::OutOfBounds {
            t_interp,
            t_prev,
            t_curr,
        },
        Error::NoLags => Error::NoLags,
        Error::InsufficientHistory {
            t_delayed,
            t_prev,
            t_curr,
        } => Error::InsufficientHistory {
            t_delayed,
            t_prev,
            t_curr,
        },
        Error::LinearAlgebra { msg } => Error::LinearAlgebra { msg },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::ExplicitRungeKutta;
    use nalgebra::{Vector1, Vector2, vector};

    struct Decay;

    impl ODE<f64, Vector1<f64>> for Decay {
        fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
            dydt[0] = -y[0];
        }
    }

    impl ODEParameters<f64, Vector1<f64>, Vector1<f64>> for Decay {
        fn diff_with_params(
            &self,
            _t: f64,
            y: &Vector1<f64>,
            p: &Vector1<f64>,
            dydt: &mut Vector1<f64>,
        ) {
            dydt[0] = -p[0] * y[0];
        }

        fn jacobian_state(
            &self,
            _t: f64,
            _y: &Vector1<f64>,
            p: &Vector1<f64>,
            jy: &mut Matrix<f64>,
        ) {
            jy[(0, 0)] = -p[0];
        }

        fn jacobian_params(
            &self,
            _t: f64,
            y: &Vector1<f64>,
            _p: &Vector1<f64>,
            jp: &mut Matrix<f64>,
        ) {
            jp[(0, 0)] = -y[0];
        }
    }

    #[test]
    fn forward_sensitivity_matches_exponential_decay_gradient() {
        let equation = Decay;
        let params = vector![0.5];
        let y0 = vector![1.0];
        let fsa = ForwardSensitivityODE::new(&equation, &params, y0);
        let y_aug0 = vector![1.0, 0.0];
        let mut method = ExplicitRungeKutta::dop853().rtol(1e-11).atol(1e-12);
        let mut solout = DefaultSolout::new();

        let solution = solve_ode(&mut method, &fsa, 0.0, 2.0, &y_aug0, &mut solout).unwrap();
        let y_final = solution.y.last().unwrap();

        let expected_y = (-1.0_f64).exp();
        let expected_s = -2.0 * expected_y;
        assert!((y_final[0] - expected_y).abs() < 1e-8);
        assert!((y_final[1] - expected_s).abs() < 1e-8);
    }

    struct TerminalState;

    impl AdjointCost<f64, Vector1<f64>, Vector1<f64>> for TerminalState {
        fn terminal(&self, _tf: f64, yf: &Vector1<f64>, _p: &Vector1<f64>) -> f64 {
            yf[0]
        }

        fn terminal_gradient_y(
            &self,
            _tf: f64,
            _yf: &Vector1<f64>,
            _p: &Vector1<f64>,
            grad_y: &mut Vector1<f64>,
        ) {
            grad_y[0] = 1.0;
        }
    }

    #[test]
    fn adjoint_sensitivity_matches_terminal_gradient() {
        let equation = Decay;
        let cost = TerminalState;
        let params = vector![0.5];
        let y0 = vector![1.0];
        let mut forward = ExplicitRungeKutta::dop853().rtol(1e-11).atol(1e-12);
        let mut backward = ExplicitRungeKutta::dop853().rtol(1e-11).atol(1e-12);

        let solution = solve_adjoint_sensitivity(
            &mut forward,
            &mut backward,
            &equation,
            &cost,
            0.0,
            2.0,
            &y0,
            &params,
        )
        .unwrap();

        let expected_y0 = (-1.0_f64).exp();
        let expected_p = -2.0 * expected_y0;
        assert!((solution.grad_y0[0] - expected_y0).abs() < 1e-6);
        assert!((solution.grad_p[0] - expected_p).abs() < 1e-6);
    }

    #[test]
    fn adjoint_state_behaves_like_flat_state() {
        let mut state = AdjointState::new(vector![1.0], Vector2::new(2.0, 3.0));
        assert_eq!(state.len(), 3);
        assert_eq!(state.get(2), 3.0);
        state.set(1, 4.0);
        assert_eq!(state.mu[0], 4.0);
    }
}
