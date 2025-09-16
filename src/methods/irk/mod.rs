//! Implicit Runge–Kutta (IRK) methods

mod adaptive;
mod fixed;
mod radau;

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    status::Status,
    tolerance::Tolerance,
    traits::{Real, State},
};

/// IRK solver core. Supports fixed/adaptive stepping and common IRK families
/// (Gauss, Radau, Lobatto).
///
/// Type params: E (equation), F (family), T (scalar), Y (state), D (callback),
/// O (order), S (stages), I (dense output terms).
pub struct ImplicitRungeKutta<
    E,
    F,
    T: Real,
    Y: State<T>,
    const O: usize,
    const S: usize,
    const I: usize,
> {
    // Initial step
    pub h0: T,

    // Current step
    h: T,

    // Current state
    t: T,
    y: Y,
    dydt: Y,

    // Previous state
    h_prev: T,
    t_prev: T,
    y_prev: Y,
    dydt_prev: Y,

    // Stage data
    k: [Y; I], // Stage derivatives
    z: [Y; S], // Stage states (z_i)

    // Butcher tableau
    c: [T; S],          // Nodes c
    a: [[T; S]; S],     // Matrix a
    b: [T; S],          // Weights b
    bh: Option<[T; S]>, // Embedded weights

    // Newton settings
    pub newton_tol: T,          // Convergence tol
    pub max_newton_iter: usize, // Max iterations per solve

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

    // Iteration tracking
    stage_jacobians: [Matrix<T>; S], // J_i at each stage
    newton_matrix: Matrix<T>,        // I - h*(A⊗J)
    rhs_newton: Vec<T>,              // Newton RHS
    delta_k_vec: Vec<T>,             // Newton solution
    jacobian_age: usize,             // Reuse counter
    stiffness_counter: usize,
    steps: usize,
    newton_iterations: usize,    // Total Newton iterations
    jacobian_evaluations: usize, // Total jacobian evaluations
    lu_decompositions: usize,    // Total LU decompositions

    // Status
    status: Status<T, Y>,

    // Method info
    order: usize,
    stages: usize,
    dense_stages: usize,

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> Default
    for ImplicitRungeKutta<E, F, T, Y, O, S, I>
{
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),
            h_prev: T::zero(),
            t_prev: T::zero(),
            y_prev: Y::zeros(),
            dydt_prev: Y::zeros(),
            k: [Y::zeros(); I],
            z: [Y::zeros(); S],
            c: [T::zero(); S],
            a: [[T::zero(); S]; S],
            b: [T::zero(); S],
            bh: None,
            newton_tol: T::from_f64(1.0e-10).unwrap(),
            max_newton_iter: 50,
            rtol: Tolerance::Scalar(T::from_f64(1.0e-6).unwrap()),
            atol: Tolerance::Scalar(T::from_f64(1.0e-6).unwrap()),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10_000,
            max_rejects: 100,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            stiffness_counter: 0,
            steps: 0,
            newton_iterations: 0,
            jacobian_evaluations: 0,
            lu_decompositions: 0,
            status: Status::Uninitialized,
            order: O,
            stages: S,
            dense_stages: I,
            family: PhantomData,
            equation: PhantomData,
            stage_jacobians: core::array::from_fn(|_| Matrix::zeros(0, 0)),
            newton_matrix: Matrix::zeros(0, 0),
            rhs_newton: Vec::new(),
            delta_k_vec: Vec::new(),
            jacobian_age: 0,
        }
    }
}

impl<E, F, T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize>
    ImplicitRungeKutta<E, F, T, Y, O, S, I>
{
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

    /// Set minimum scale for step changes (default: 0.2)
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

    /// Set max Newton iterations per stage (default: 50)
    pub fn max_newton_iter(mut self, max_newton_iter: usize) -> Self {
        self.max_newton_iter = max_newton_iter;
        self
    }

    /// Get method order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get number of stages
    pub fn stages(&self) -> usize {
        self.stages
    }

    /// Get dense output terms
    pub fn dense_stages(&self) -> usize {
        self.dense_stages
    }
}
