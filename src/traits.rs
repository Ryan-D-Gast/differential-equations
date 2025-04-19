//! Defines Generics for the library. Includes generics for the floating point numbers.

use std::ops::{Add, AddAssign, Div, Mul, Sub};
use nalgebra::{SMatrix, RealField};

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
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    fn infinity() -> Self {
        std::f32::INFINITY
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    fn infinity() -> Self {
        std::f64::INFINITY
    }

    fn to_f64(self) -> f64 {
        self
    }
}

/// State vector trait
pub trait State<T>:
    Clone +
    Copy +
    Add<Output = Self> +
    Sub<Output = Self> +
    AddAssign +
    Mul<T, Output = Self> +
    Div<T, Output = Self> +
{
    fn len(&self) -> usize;

    fn get(&self, i: usize) -> T; 

    fn zeros() -> Self;
}

impl<T: Real> State<T> for T
{
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
        if i < self.len() {
            self[(i / C, i % C)]
        } else {
            panic!("Index out of bounds")
        }
    }

    fn zeros() -> Self {
        SMatrix::<T, R, C>::zeros()
    }
}

// Linear Algebra helper function for trait
pub(crate) fn dot<T, V>(
    a: &V,
    b: &V,
) -> T
where
    T: Real,
    V: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * b.get(i);
    }
    sum
}

pub(crate) fn norm<T, V>(
    a: V,
) -> T
where
    T: Real,
    V: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * a.get(i);
    }
    sum.sqrt()
}

/// Callback data trait
///
/// This trait represents data that can be returned from functions
/// that are used to control the solver's execution flow. The
/// Clone and Debug traits are required for internal use but anything
/// that implements this trait can be used as callback data.
/// For example, this can be a string, a number, or any other type
/// that implements the Clone and Debug traits.
///
pub trait CallBackData: Clone + std::fmt::Debug {}

// Implement for any type that already satisfies the bounds
impl<T: Clone + std::fmt::Debug> CallBackData for T {}