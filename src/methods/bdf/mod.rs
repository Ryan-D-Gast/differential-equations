//! Backward Differentiation Formula (BDF) methods

mod fixed;

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    status::Status,
    tolerance::Tolerance,
    traits::{Real, State},
};

pub struct BackwardDifferentiationFormula<
    E,
    F,
    T: Real,
    Y: State<T>,
    const O: usize,
> {
    // Initial step
    pub h0: T,

    // Current step
    h: T,

    // Current state
    t: T,
    y: Y,
    dydt: Y,

    // Previous states for history
    t_prev: [T; O],
    y_prev: [Y; O],

    // Alpha coefficients for the derivative approximation
    alpha: [T; O],
    // Gamma coefficients for Newton step
    gamma: T,

    // Newton settings
    pub newton_tol: T,
    pub max_newton_iter: usize,

    // Adaptive settings
    pub rtol: Tolerance<T>,
    pub atol: Tolerance<T>,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub max_rejects: usize,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,
    pub filter: fn(T) -> T,

    // Tracking
    jacobian: Matrix<T>,
    newton_matrix: Matrix<T>,
    rhs_newton: Vec<T>,
    delta_y_vec: Vec<T>,
    steps: usize,
    jacobian_evaluations: usize,
    lu_decompositions: usize,
    jacobian_age: usize,

    // Status
    status: Status<T, Y>,

    // Method info
    order: usize,

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,
}

impl<E, F, T: Real, Y: State<T>, const O: usize> Default
    for BackwardDifferentiationFormula<E, F, T, Y, O>
{
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),
            t_prev: [T::zero(); O],
            y_prev: [Y::zeros(); O],
            alpha: [T::zero(); O],
            gamma: T::zero(),
            newton_tol: T::from_f64(1.0e-10).unwrap(),
            max_newton_iter: 20,
            rtol: Tolerance::Scalar(T::from_f64(1.0e-6).unwrap()),
            atol: Tolerance::Scalar(T::from_f64(1.0e-9).unwrap()),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10_000,
            max_rejects: 10,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.1).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            filter: |h| h,
            jacobian: Matrix::zeros(0, 0),
            newton_matrix: Matrix::zeros(0, 0),
            rhs_newton: Vec::new(),
            delta_y_vec: Vec::new(),
            steps: 0,
            jacobian_evaluations: 0,
            lu_decompositions: 0,
            jacobian_age: 0,
            status: Status::Uninitialized,
            order: O,
            family: PhantomData,
            equation: PhantomData,
        }
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize> BackwardDifferentiationFormula<E, F, T, Y, O> {
    /// Set relative tolerance
    pub fn rtol<V: Into<Tolerance<T>>>(mut self, rtol: V) -> Self {
        self.rtol = rtol.into();
        self
    }

    /// Set absolute tolerance
    pub fn atol<V: Into<Tolerance<T>>>(mut self, atol: V) -> Self {
        self.atol = atol.into();
        self
    }

    /// Set initial step size
    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    /// Set minimum step size
    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set maximum step size
    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    /// Set max steps
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set max consecutive rejections
    pub fn max_rejects(mut self, max_rejects: usize) -> Self {
        self.max_rejects = max_rejects;
        self
    }

    /// Set step size safety factor (default: 0.9)
    pub fn safety_factor(mut self, safety_factor: T) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    /// Set minimum scale for step changes (default: 0.1)
    pub fn min_scale(mut self, min_scale: T) -> Self {
        self.min_scale = min_scale;
        self
    }

    /// Set maximum scale for step changes (default: 10.0)
    pub fn max_scale(mut self, max_scale: T) -> Self {
        self.max_scale = max_scale;
        self
    }

    /// Set Newton tolerance (default: 1e-10)
    pub fn newton_tol(mut self, newton_tol: T) -> Self {
        self.newton_tol = newton_tol;
        self
    }

    /// Set max Newton iterations per stage (default: 20)
    pub fn max_newton_iter(mut self, max_newton_iter: usize) -> Self {
        self.max_newton_iter = max_newton_iter;
        self
    }

    /// Set the step size filter (default: identity)
    pub fn filter(mut self, filter: fn(T) -> T) -> Self {
        self.filter = filter;
        self
    }

    /// Get method order
    pub fn order(&self) -> usize {
        self.order
    }
}
