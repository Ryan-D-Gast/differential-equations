//! Adams Predictor Correction (APC) methods

mod apcf4;
mod apcv4;

use crate::{
    Status,
    traits::{CallBackData, Real, State},
};
use std::marker::PhantomData;

pub struct AdamsPredictorCorrector<E, F, T: Real, V: State<T>, D: CallBackData, const S: usize> {
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: V,
    dydt: V,

    // Final Time
    tf: T,

    // Previous States
    t_prev: [T; S],
    y_prev: [V; S],

    // Previous step states
    t_old: T,
    y_old: V,
    dydt_old: V,

    // Predictor Correct Derivatives
    k: [V; S],

    // Statistic Tracking
    evals: usize,
    steps: usize,

    // Status
    status: Status<T, V, D>,

    // Settings
    pub tol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,

    // Method info
    pub stages: usize,

    // Family classification
    family: PhantomData<F>,

    // Equation type
    equation: PhantomData<E>,
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const S: usize> Default
    for AdamsPredictorCorrector<E, F, T, V, D, S>
{
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: V::zeros(),
            dydt: V::zeros(),
            t_prev: [T::zero(); S],
            y_prev: [V::zeros(); S],
            t_old: T::zero(),
            y_old: V::zeros(),
            dydt_old: V::zeros(),
            k: [V::zeros(); S],
            tf: T::zero(),
            evals: 0,
            steps: 0,
            status: Status::Uninitialized,
            tol: T::from_f64(1.0e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10_000,
            stages: S,
            family: PhantomData,
            equation: PhantomData,
        }
    }
}

impl<E, F, T: Real, V: State<T>, D: CallBackData, const S: usize>
    AdamsPredictorCorrector<E, F, T, V, D, S>
{
    /// Set the tolerance for error control
    pub fn tol(mut self, rtol: T) -> Self {
        self.tol = rtol;
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

    /// Get the number of stages in the method
    pub fn stages(&self) -> usize {
        self.stages
    }
}
