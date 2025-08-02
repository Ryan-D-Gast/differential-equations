//! Statistics and performance tracking for Numerical methods

use crate::traits::Real;
use std::{
    ops::{Add, AddAssign},
    time::Instant,
};

/// Number of evaluations
///
/// # Fields
/// * `function` - Number of function evaluations
/// * `jacobian` - Number of jacobian evaluations
/// * `newton` - Number of Newton iterations
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Evals {
    pub function: usize,
    pub jacobian: usize,
    pub newton: usize,
}

impl Evals {
    /// Create a new Evals struct
    ///
    /// # Arguments
    /// * `diff` - Number of differntial equation function evaluations
    /// * `jacobian`  - Number of jacobian evaluations
    /// * `newton` - Number of Newton iterations
    pub fn new() -> Self {
        Self {
            function: 0,
            jacobian: 0,
            newton: 0,
        }
    }
}

impl Add for Evals {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            function: self.function + other.function,
            jacobian: self.jacobian + other.jacobian,
            newton: self.newton + other.newton,
        }
    }
}

impl AddAssign for Evals {
    fn add_assign(&mut self, other: Self) {
        self.function += other.function;
        self.jacobian += other.jacobian;
        self.newton += other.newton;
    }
}

/// Number of Steps
///
/// # Fields
/// * `accepted` - Number of accepted steps
/// * `rejected` - Number of rejected steps
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Steps {
    pub accepted: usize,
    pub rejected: usize,
}

impl Steps {
    /// Create a new Steps struct
    pub fn new() -> Self {
        Self {
            accepted: 0,
            rejected: 0,
        }
    }

    /// Get the total number of steps (accepted + rejected)
    pub fn total(&self) -> usize {
        self.accepted + self.rejected
    }
}

impl Add for Steps {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            accepted: self.accepted + other.accepted,
            rejected: self.rejected + other.rejected,
        }
    }
}

impl AddAssign for Steps {
    fn add_assign(&mut self, other: Self) {
        self.accepted += other.accepted;
        self.rejected += other.rejected;
    }
}

/// Timer for tracking solution time
#[derive(Debug, Clone)]
pub enum Timer<T: Real> {
    Off,
    Running(Instant),
    Completed(T),
}

impl<T: Real> Timer<T> {
    /// Starts the timer
    pub fn start(&mut self) {
        *self = Timer::Running(Instant::now());
    }

    /// Returns the elapsed time in seconds
    pub fn elapsed(&self) -> T {
        match self {
            Timer::Off => T::zero(),
            Timer::Running(start_time) => T::from_f64(start_time.elapsed().as_secs_f64()).unwrap(),
            Timer::Completed(t) => *t,
        }
    }

    /// Complete the running timer and convert it to a completed state
    pub fn complete(&mut self) {
        match self {
            Timer::Off => {}
            Timer::Running(start_time) => {
                *self = Timer::Completed(T::from_f64(start_time.elapsed().as_secs_f64()).unwrap());
            }
            Timer::Completed(_) => {}
        }
    }
}
