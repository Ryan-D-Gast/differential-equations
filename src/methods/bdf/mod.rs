//! Adaptive variable-order Backward Differentiation Formula (BDF) solver.
//!
//! Implements BDF orders 1 through 5 with automatic order selection and
//! adaptive step size control. The solver starts at order 1 and adjusts
//! both order and step size to balance accuracy and efficiency.
//!
//! Based on the approach described in:
//! - Hairer, E., & Wanner, G. (1996). Solving Ordinary Differential
//!   Equations II, Section IV.2 (BDF methods).

mod adaptive;

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    status::Status,
    tolerance::Tolerance,
    traits::{Real, State},
};

/// Maximum BDF order (BDF5 is the highest practically useful order).
pub const MAX_ORDER: usize = 5;

/// Adaptive variable-order Backward Differentiation Formula solver.
///
/// # Overview
/// - Solves stiff ODEs using BDF orders 1-5 with automatic order and step
///   size selection.
/// - Uses Newton iteration to resolve the implicit BDF equations.
/// - Supports dense output via polynomial interpolation.
///
/// # Order selection
/// The solver starts at order 1 and considers increasing the order after
/// enough successful steps at the current order. On step rejection, the
/// order is decreased to improve stability.
pub struct Bdf<E, T: Real, Y: State<T>> {
    // Configuration
    pub h0: T,
    pub rtol: Tolerance<T>,
    pub atol: Tolerance<T>,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,
    pub filter: fn(T) -> T,
    pub newton_tol: T,
    pub max_newton_iter: usize,
    pub max_order: usize,

    // Step size
    h: T,

    // State
    t: T,
    y: Y,
    dydt: Y,

    // History
    t_hist: [T; MAX_ORDER],
    y_hist: [Y; MAX_ORDER],
    n_hist: usize,

    // Order control
    order: usize,
    steps_at_order: usize,

    // Workspace
    jacobian: Matrix<T>,
    newton_matrix: Matrix<T>,
    rhs: Vec<T>,
    ip: Vec<usize>,
    jacobian_age: usize,

    // Error control
    err_old: T,
    reject: bool,

    // Status
    status: Status<T, Y>,
    steps: usize,

    _phantom: PhantomData<E>,
}

impl<E, T: Real, Y: State<T>> Default for Bdf<E, T, Y> {
    fn default() -> Self {
        Self {
            h0: T::zero(),
            rtol: Tolerance::Scalar(T::from_f64(1e-6).unwrap()),
            atol: Tolerance::Scalar(T::from_f64(1e-9).unwrap()),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10_000,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            filter: |h| h,
            newton_tol: T::from_f64(1e-10).unwrap(),
            max_newton_iter: 20,
            max_order: MAX_ORDER,

            h: T::zero(),
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),

            t_hist: [T::zero(); MAX_ORDER],
            y_hist: [Y::zeros(); MAX_ORDER],
            n_hist: 0,

            order: 1,
            steps_at_order: 0,

            jacobian: Matrix::zeros(0, 0),
            newton_matrix: Matrix::zeros(0, 0),
            rhs: Vec::new(),
            ip: Vec::new(),
            jacobian_age: usize::MAX,

            err_old: T::from_f64(1e-4).unwrap(),
            reject: false,

            status: Status::Uninitialized,
            steps: 0,

            _phantom: PhantomData,
        }
    }
}

impl<E, T: Real, Y: State<T>> Bdf<E, T, Y> {
    pub fn builder() -> Self {
        Self::default()
    }

    pub fn rtol<V: Into<Tolerance<T>>>(mut self, rtol: V) -> Self {
        self.rtol = rtol.into();
        self
    }

    pub fn atol<V: Into<Tolerance<T>>>(mut self, atol: V) -> Self {
        self.atol = atol.into();
        self
    }

    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn safety_factor(mut self, safety_factor: T) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    pub fn newton_tol(mut self, newton_tol: T) -> Self {
        self.newton_tol = newton_tol;
        self
    }

    pub fn max_newton_iter(mut self, max_newton_iter: usize) -> Self {
        self.max_newton_iter = max_newton_iter;
        self
    }

    pub fn max_order(mut self, max_order: usize) -> Self {
        self.max_order = max_order.clamp(1, MAX_ORDER);
        self
    }

    pub fn filter(mut self, filter: fn(T) -> T) -> Self {
        self.filter = filter;
        self
    }

    pub fn order(&self) -> usize {
        self.order
    }
}
