//! Diagonally Implicit Runge–Kutta (DIRK) methods

mod adaptive;
mod fixed;

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    status::Status,
    traits::{CallBackData, Real, State},
};

/// DIRK/SDIRK core with fixed/adaptive stepping. Stages are solved
/// sequentially (A is lower-triangular), which is cheaper than full IRK
/// while retaining strong stability for stiff problems.
///
/// Type params: E (equation), F (family), T (scalar), Y (state), D (callback),
/// O (order), S (stages), I (dense output terms).
pub struct DiagonallyImplicitRungeKutta<
    E,
    F,
    T: Real,
    Y: State<T>,
    D: CallBackData,
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

    // Stage data (solved one at a time)
    k: [Y; I], // Stage derivatives
    z: Y,      // Current stage state

    // Butcher tableau
    c: [T; S],          // Nodes c
    a: [[T; S]; S],     // Matrix a (lower-triangular)
    b: [T; S],          // Weights b
    bh: Option<[T; S]>, // Embedded weights

    // Tolerances
    pub rtol: T,
    pub atol: T,

    // Step-size control
    pub h_min: T,
    pub h_max: T,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,

    // Newton parameters
    pub newton_tol: T,
    pub max_newton_iter: usize,

    // Adaptive step control
    pub max_steps: usize,
    pub max_rejects: usize,

    // Statistics
    steps: usize,
    stiffness_counter: usize,
    newton_iterations: usize,
    jacobian_evaluations: usize,
    lu_decompositions: usize,

    // Newton workspace (per stage)
    jacobian: Matrix<T>,
    rhs_newton: Y,
    delta_z: Y,
    jacobian_age: usize,

    // Status
    status: Status<T, Y, D>,

    // Method info
    order: usize,
    stages: usize,

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,
}

impl<E, F, T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Default for DiagonallyImplicitRungeKutta<E, F, T, Y, D, O, S, I>
{
    fn default() -> Self {
        let dim = 1; // Will be resized during init
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
            z: Y::zeros(),
            c: [T::zero(); S],
            a: [[T::zero(); S]; S],
            b: [T::zero(); S],
            bh: None,
            rtol: T::from_f64(1e-6).unwrap(),
            atol: T::from_f64(1e-9).unwrap(),
            h_min: T::from_f64(1e-14).unwrap(),
            h_max: T::from_f64(1e3).unwrap(),
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.1).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            newton_tol: T::from_f64(1e-10).unwrap(),
            max_newton_iter: 20,
            max_steps: 10_000,
            max_rejects: 10,
            steps: 0,
            stiffness_counter: 0,
            newton_iterations: 0,
            jacobian_evaluations: 0,
            lu_decompositions: 0,
            jacobian: Matrix::zeros(dim),
            rhs_newton: Y::zeros(),
            delta_z: Y::zeros(),
            jacobian_age: 0,
            status: Status::Uninitialized,
            order: O,
            stages: S,
            family: PhantomData,
            equation: PhantomData,
        }
    }
}

// Builder methods
impl<E, F, T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    DiagonallyImplicitRungeKutta<E, F, T, Y, D, O, S, I>
{
    /// Set relative tolerance
    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set absolute tolerance
    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
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

    /// Get method order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get number of stages
    pub fn stages(&self) -> usize {
        self.stages
    }
}
