//! Implicit Runge-Kutta (IRK) methods

mod adaptive;
mod fixed;

use crate::{
    Status,
    traits::{CallBackData, Real, State},
};
use std::{
    marker::PhantomData
};

/// Implicit Runge-Kutta solver that can handle:
/// - Fixed-step methods with Newton iteration for stage equations
/// - Adaptive step methods with embedded error estimation
/// - Gauss methods (A-stable, symplectic for Hamiltonian systems)
/// - Radau methods (L-stable, good for stiff problems)
/// - Lobatto methods (A-stable, good for constrained systems)
///
/// # Type Parameters
///
/// * `E`: Equation type (e.g., Ordinary, Delay, Stochastic)
/// * `F`: Family type (e.g., Adaptive, Fixed, Gauss, Radau, Lobatto)
/// * `T`: Real number type (f32, f64)
/// * `V`: State vector type
/// * `D`: Callback data type
/// * `const O`: Order of the method
/// * `const S`: Number of stages in the method
/// * `const I`: Total number of stages including interpolation (equal to S for methods without dense output)
pub struct ImplicitRungeKutta<E, F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> {
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: V,
    dydt: V,

    // Previous State
    h_prev: T,
    t_prev: T,
    y_prev: V,
    dydt_prev: V,

    // Stage values
    k: [V; I],           // Stage derivatives
    y_stages: [V; S],    // Stage values (Y_i = y_n + h * sum(a_ij * k_j))

    // Constants from Butcher tableau
    c: [T; S],                    // Stage time coefficients
    a: [[T; S]; S],              // Runge-Kutta matrix (typically dense for implicit methods)
    b: [T; S],                   // Weights for final solution
    bh: Option<[T; S]>,          // Lower order coefficients for embedded methods

    // Newton iteration settings
    pub newton_tol: T,           // Tolerance for Newton iteration convergence
    pub max_newton_iter: usize,  // Maximum Newton iterations per stage

    // Settings
    pub rtol: T,
    pub atol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub max_rejects: usize,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,

    // Iteration tracking
    stage_jacobians: [nalgebra::DMatrix<T>; S], // Stage-specific Jacobian matrices J_i = df/dy(t + c_i*h, z_i)
    newton_matrix: nalgebra::DMatrix<T>,      // Newton system matrix M = I - h*(A⊗J)
    rhs_newton: nalgebra::DVector<T>,         // Right-hand side vector for Newton system
    delta_k_vec: nalgebra::DVector<T>,        // Solution vector for Newton system
    jacobian_age: usize,                      // Age of current Jacobian (for reuse)
    stiffness_counter: usize,
    steps: usize,
    newton_iterations: usize,    // Total Newton iterations
    jacobian_evaluations: usize, // Total Jacobian evaluations
    lu_decompositions: usize,    // Total LU decompositions

    // Status
    status: Status<T, V, D>,
    
    // Method info
    order: usize,
    stages: usize,
    dense_stages: usize,

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Default for ImplicitRungeKutta<E, F, T, V, D, O, S, I> {
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: V::zeros(),
            dydt: V::zeros(),
            h_prev: T::zero(),
            t_prev: T::zero(),
            y_prev: V::zeros(),
            dydt_prev: V::zeros(),
            k: [V::zeros(); I],
            y_stages: [V::zeros(); S],
            c: [T::zero(); S],
            a: [[T::zero(); S]; S],
            b: [T::zero(); S],
            bh: None,
            newton_tol: T::from_f64(1.0e-10).unwrap(),
            max_newton_iter: 50,
            rtol: T::from_f64(1.0e-6).unwrap(),
            atol: T::from_f64(1.0e-6).unwrap(),
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
            stage_jacobians: core::array::from_fn(|_| nalgebra::DMatrix::zeros(0, 0)),
            newton_matrix: nalgebra::DMatrix::zeros(0, 0),
            rhs_newton: nalgebra::DVector::zeros(0),
            delta_k_vec: nalgebra::DVector::zeros(0),
            jacobian_age: 0,
        }
    }
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> ImplicitRungeKutta<E, F, T, V, D, O, S, I> {
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

    /// Set the minimum scale factor for step size changes (default: 0.2)
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

    /// Set the maximum number of Newton iterations per stage (default: 50)
    pub fn max_newton_iter(mut self, max_newton_iter: usize) -> Self {
        self.max_newton_iter = max_newton_iter;
        self
    }

    /// Get the order of the method
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the number of stages in the method
    pub fn stages(&self) -> usize {
        self.stages
    }

    /// Get the number of terms in the dense output interpolation polynomial
    pub fn dense_stages(&self) -> usize {
        self.dense_stages
    }
}