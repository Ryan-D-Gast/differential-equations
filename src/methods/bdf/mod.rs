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
const BDF_ROWS: usize = MAX_ORDER + 3;

/// Adaptive variable-order Backward Differentiation Formula solver.
///
/// Uses BDF orders 1-5 with automatic order and step size selection.
/// Newton iteration resolves the implicit BDF equations. Dense output
/// via polynomial interpolation is supported.
///
/// The solver starts at order 1 and considers increasing the order after
/// enough successful steps. On step rejection, the order is decreased.
pub struct BDF<E, T: Real, Y: State<T>> {
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

    h: T,
    t: T,
    y: Y,
    dydt: Y,

    t_prev: T,
    y_prev: Y,
    h_prev: T,
    d: [Y; BDF_ROWS],

    order: usize,
    n_equal_steps: usize,

    gamma: [T; MAX_ORDER + 1],
    alpha: [T; MAX_ORDER + 1],
    error_const: [T; MAX_ORDER + 2],

    jacobian: Matrix<T>,
    newton_matrix: Matrix<T>,
    ip: Vec<usize>,
    lu_valid: bool,

    status: Status<T, Y>,
    steps: usize,

    _phantom: PhantomData<E>,
}

impl<E, T: Real, Y: State<T>> Default for BDF<E, T, Y> {
    fn default() -> Self {
        Self {
            h0: T::zero(),
            rtol: Tolerance::Scalar(T::from_f64(1e-8).unwrap()),
            atol: Tolerance::Scalar(T::from_f64(1e-10).unwrap()),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10_000,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            filter: |h| h,
            newton_tol: T::zero(),
            max_newton_iter: 4,
            max_order: MAX_ORDER,

            h: T::zero(),
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),

            t_prev: T::zero(),
            y_prev: Y::zeros(),
            h_prev: T::zero(),
            d: [Y::zeros(); BDF_ROWS],

            order: 1,
            n_equal_steps: 0,

            gamma: [T::zero(); MAX_ORDER + 1],
            alpha: [T::zero(); MAX_ORDER + 1],
            error_const: [T::zero(); MAX_ORDER + 2],

            jacobian: Matrix::zeros(0, 0),
            newton_matrix: Matrix::zeros(0, 0),
            ip: Vec::new(),
            lu_valid: false,

            status: Status::Uninitialized,
            steps: 0,

            _phantom: PhantomData,
        }
    }
}

impl<E, T: Real, Y: State<T>> BDF<E, T, Y> {
    pub fn adaptive() -> Self {
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
