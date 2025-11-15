//! Statistics and performance tracking for Numerical methods

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{
    ops::{Add, AddAssign},
    time::Instant,
};

use crate::traits::Real;

/// Number of evaluations
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Evals {
    /// Number of function evaluations
    pub function: usize,
    /// Number of jacobian evaluations
    pub jacobian: usize,
    /// Number of Newton iterations
    pub newton: usize,
    /// Number of matrix decompositions
    pub decompositions: usize,
    /// Number of matrix system solves
    pub solves: usize,
}

impl Evals {
    /// Create a new Evals struct with zeroed fields
    pub fn new() -> Self {
        Self {
            function: 0,
            jacobian: 0,
            newton: 0,
            decompositions: 0,
            solves: 0,
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
            decompositions: self.decompositions + other.decompositions,
            solves: self.solves + other.solves,
        }
    }
}

impl AddAssign for Evals {
    fn add_assign(&mut self, other: Self) {
        self.function += other.function;
        self.jacobian += other.jacobian;
        self.newton += other.newton;
        self.decompositions += other.decompositions;
        self.solves += other.solves;
    }
}

/// Number of Steps
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Steps {
    /// Number of accepted steps
    pub accepted: usize,
    /// Number of rejected steps
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Timer<T: Real> {
    Off,
    #[cfg_attr(feature = "serde", serde(skip_serializing, skip_deserializing))]
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
