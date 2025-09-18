//! Adams Predictor Correction (APC) methods

mod apcf4;
mod apcv4;

use std::marker::PhantomData;

use crate::{
    status::Status,
    traits::{Real, State},
};

pub struct AdamsPredictorCorrector<E, F, T: Real, Y: State<T>, const S: usize> {
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: Y,
    dydt: Y,

    // Final Time
    tf: T,

    // Previous States
    t_prev: [T; S],
    y_prev: [Y; S],

    // Previous step states
    t_old: T,
    y_old: Y,
    dydt_old: Y,

    // Predictor Correct Derivatives
    k: [Y; S],

    // Statistic Tracking
    evals: usize,
    steps: usize,

    // Status
    status: Status<T, Y>,

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

impl<E, F, T: Real, Y: State<T>, const S: usize> Default
    for AdamsPredictorCorrector<E, F, T, Y, S>
{
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),
            t_prev: [T::zero(); S],
            y_prev: [Y::zeros(); S],
            t_old: T::zero(),
            y_old: Y::zeros(),
            dydt_old: Y::zeros(),
            k: [Y::zeros(); S],
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

impl<E, F, T: Real, Y: State<T>, const S: usize> AdamsPredictorCorrector<E, F, T, Y, S> {
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
