//! Explicit Runge-Kutta (ERK) methods

mod adaptive;
mod dormandprince;
mod fixed;

use crate::{
    Status,
    methods::Delay,
    traits::{CallBackData, Real, State},
};
use std::{collections::VecDeque, marker::PhantomData};

/// Runge-Kutta solver that can handle:
/// - Fixed-step methods with cubic Hermite interpolation
/// - Adaptive step methods with embedded error estimation and cubic Hermite interpolation
/// - Advanced methods with dense output interpolation using Butcher tableau coefficients
///
/// # Type Parameters
///
/// * `E`: Equation type (e.g., Ordinary, Delay, Stochastic)
/// * `F`: Family type (e.g., Adaptive, Fixed, DormandPrince, etc.)
/// * `T`: Real number type (f32, f64)
/// * `V`: State vector type
/// * `D`: Callback data type
/// * `const O`: Order of the method
/// * `const S`: Number of stages in the method
/// * `const I`: Total number of stages including interpolation (equal to S for methods without dense output)
pub struct ExplicitRungeKutta<
    E,
    F,
    T: Real,
    V: State<T>,
    D: CallBackData,
    const O: usize,
    const S: usize,
    const I: usize,
> {
    // Domain of problem
    t0: T,

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
    k: [V; I],

    // Constants from Butcher tableau
    c: [T; I],
    a: [[T; I]; I],
    b: [T; S],
    bh: Option<[T; S]>, // Lower order coefficients for embedded methods
    er: Option<[T; S]>, // Error estimation coefficients

    // Interpolation coefficients
    bi: Option<[[T; I]; I]>, // Optional for methods without dense output
    cont: [V; O],

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
    stiffness_counter: usize,
    non_stiffness_counter: usize,
    steps: usize,

    // Status
    status: Status<T, V, D>,

    // Method info
    order: usize,
    stages: usize,
    dense_stages: usize,
    fsal: bool, // First Same As Last (FSAL) property

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,

    // DDE (Delay) support
    history: VecDeque<(T, V, V)>, // (t, y, dydt)
    max_delay: Option<T>,         // Minimum delay for DDEs so that buffer can be emptied
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Default for ExplicitRungeKutta<E, F, T, V, D, O, S, I>
{
    fn default() -> Self {
        Self {
            t0: T::zero(),
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
            c: [T::zero(); I],
            a: [[T::zero(); I]; I],
            b: [T::zero(); S],
            bh: None,
            er: None,
            bi: None,
            cont: [V::zeros(); O],
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
            non_stiffness_counter: 0,
            steps: 0,
            status: Status::Uninitialized,
            order: O,
            stages: S,
            dense_stages: I,
            fsal: false,
            family: PhantomData,
            equation: PhantomData,
            history: VecDeque::new(),
            max_delay: None,
        }
    }
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    ExplicitRungeKutta<E, F, T, V, D, O, S, I>
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

impl<F, T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    ExplicitRungeKutta<Delay, F, T, V, D, O, S, I>
{
    /// Set the maximum delay for DDEs
    pub fn max_delay(mut self, max_delay: T) -> Self {
        self.max_delay = Some(max_delay);
        self
    }
}
