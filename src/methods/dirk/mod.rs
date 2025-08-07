//! Diagonally Implicit Runge-Kutta (DIRK) methods

mod adaptive;
mod fixed;

use crate::{
    Status,
    traits::{CallBackData, Real, State},
};
use std::marker::PhantomData;

/// Diagonally Implicit Runge-Kutta solver that can handle:
/// - Fixed-step methods with Newton iteration for each stage
/// - Adaptive step methods with embedded error estimation
/// - DIRK methods (L-stable, good for stiff problems)
/// - SDIRK methods (singly diagonally implicit, all diagonal entries equal)
///
/// Key difference from IRK: DIRK methods solve stages sequentially since
/// the coefficient matrix A is lower triangular, making them more efficient
/// than fully implicit methods while maintaining good stability properties.
///
/// # Type Parameters
///
/// * `E`: Equation type (e.g., Ordinary, Delay, Stochastic)
/// * `F`: Family type (e.g., Adaptive, Fixed)
/// * `T`: Real number type (f32, f64)
/// * `Y`: State vector type
/// * `D`: Callback data type
/// * `const O`: Order of the method
/// * `const S`: Number of stages in the method
/// * `const I`: Number of dense output stages (for interpolation)
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
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: Y,
    dydt: Y,

    // Previous State
    h_prev: T,
    t_prev: T,
    y_prev: Y,
    dydt_prev: Y,

    // Stage values - DIRK solves one stage at a time
    k: [Y; I],  // Stage derivatives
    z_stage: Y, // Current stage solution being solved

    // Constants from Butcher tableau
    c: [T; S],          // Stage time coefficients
    a: [[T; S]; S],     // Coefficient matrix (lower triangular for DIRK)
    b: [T; S],          // Solution weights
    bh: Option<[T; S]>, // Lower order weights for embedded methods

    // Tolerances for adaptive stepping
    pub rtol: T,
    pub atol: T,

    // Step size control
    pub h_min: T,
    pub h_max: T,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,

    // Newton solver parameters
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

    // Linear algebra workspace for Newton solver (per stage)
    stage_jacobian: nalgebra::DMatrix<T>,
    newton_matrix: nalgebra::DMatrix<T>, // I - h*a_ii*J for current stage
    rhs_newton: nalgebra::DVector<T>,    // Newton RHS for current stage
    delta_z: nalgebra::DVector<T>,       // Newton correction for current stage
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
            z_stage: Y::zeros(),
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
            stage_jacobian: nalgebra::DMatrix::zeros(dim, dim),
            newton_matrix: nalgebra::DMatrix::zeros(dim, dim),
            rhs_newton: nalgebra::DVector::zeros(dim),
            delta_z: nalgebra::DVector::zeros(dim),
            jacobian_age: 0,
            status: Status::Uninitialized,
            order: O,
            stages: S,
            family: PhantomData,
            equation: PhantomData,
        }
    }
}

// Builder methods for configuration
impl<E, F, T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    DiagonallyImplicitRungeKutta<E, F, T, Y, D, O, S, I>
{
    /// Set the relative tolerance for error control
    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set the absolute tolerance for error control
    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }

    /// Set the initial step size
    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    /// Set the minimum allowed step size
    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set the maximum allowed step size
    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    /// Set the maximum number of steps allowed
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set the maximum number of consecutive rejected steps before declaring stiffness
    pub fn max_rejects(mut self, max_rejects: usize) -> Self {
        self.max_rejects = max_rejects;
        self
    }

    /// Set the safety factor for step size control (default: 0.9)
    pub fn safety_factor(mut self, safety_factor: T) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    /// Set the minimum scale factor for step size changes (default: 0.1)
    pub fn min_scale(mut self, min_scale: T) -> Self {
        self.min_scale = min_scale;
        self
    }

    /// Set the maximum scale factor for step size changes (default: 10.0)
    pub fn max_scale(mut self, max_scale: T) -> Self {
        self.max_scale = max_scale;
        self
    }

    /// Set the Newton iteration tolerance (default: 1e-10)
    pub fn newton_tol(mut self, newton_tol: T) -> Self {
        self.newton_tol = newton_tol;
        self
    }

    /// Set the maximum number of Newton iterations per stage (default: 20)
    pub fn max_newton_iter(mut self, max_newton_iter: usize) -> Self {
        self.max_newton_iter = max_newton_iter;
        self
    }

    /// Get the order of the method
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the number of stages
    pub fn stages(&self) -> usize {
        self.stages
    }
}
