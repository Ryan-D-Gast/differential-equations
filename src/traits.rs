//! Defines Generics for the library. Includes generics for the floating point numbers.

use nalgebra::{RealField, SMatrix};
use num_complex::Complex;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

/// Real Number Trait
///
/// This trait specifies the acceptable types for real numbers.
/// Currently implemented for:
/// * `f32` - 32-bit floating point
/// * `f64` - 64-bit floating point
///
/// Provides additional functionality required for ODE solvers beyond
/// what's provided by nalgebra's RealField trait.
///
pub trait Real: Copy + RealField {
    fn infinity() -> Self;
}

impl Real for f32 {
    fn infinity() -> Self {
        f32::INFINITY
    }
}

impl Real for f64 {
    fn infinity() -> Self {
        f64::INFINITY
    }
}

/// State vector trait
///
/// Represents the state of the system being solved.
///
/// Implements for the following types:
/// * `f32` - 32-bit floating point
/// * `f64` - 64-bit floating point
/// * `SMatrix` - Matrix type from nalgebra
/// * `Complex` - Complex number type from num-complex
/// * `Struct<T>` - Any struct with all fields of type T using #[derive(State)] from the `derive` module
///
pub trait State<T: Real>:
    Clone
    + Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + Neg<Output = Self>
{
    fn len(&self) -> usize;

    fn get(&self, i: usize) -> T;

    fn set(&mut self, i: usize, value: T);

    fn zeros() -> Self;
}

impl<T: Real> State<T> for T {
    fn len(&self) -> usize {
        1
    }

    fn get(&self, i: usize) -> T {
        if i == 0 {
            *self
        } else {
            panic!("Index out of bounds")
        }
    }

    fn set(&mut self, i: usize, value: T) {
        if i == 0 {
            *self = value;
        } else {
            panic!("Index out of bounds")
        }
    }

    fn zeros() -> Self {
        T::zero()
    }
}

impl<T, const R: usize, const C: usize> State<T> for SMatrix<T, R, C>
where
    T: Real,
{
    fn len(&self) -> usize {
        R * C
    }

    fn get(&self, i: usize) -> T {
        self[(i / C, i % C)]
    }

    fn set(&mut self, i: usize, value: T) {
        self[(i / C, i % C)] = value;
    }

    fn zeros() -> Self {
        SMatrix::<T, R, C>::zeros()
    }
}

impl<T> State<T> for Complex<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        2
    }

    fn get(&self, i: usize) -> T {
        if i == 0 {
            self.re
        } else if i == 1 {
            self.im
        } else {
            panic!("Index out of bounds")
        }
    }

    fn set(&mut self, i: usize, value: T) {
        if i == 0 {
            self.re = value;
        } else if i == 1 {
            self.im = value;
        } else {
            panic!("Index out of bounds")
        }
    }

    fn zeros() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}
