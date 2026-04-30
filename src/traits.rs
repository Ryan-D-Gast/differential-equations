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

impl<T: Copy + RealField> Real for T {
    #[inline]
    fn infinity() -> Self {
        Self::from_subset(&f64::INFINITY)
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

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, i: usize) -> T;

    fn set(&mut self, i: usize, value: T);

    fn zeros() -> Self;
}

impl<T: Real> State<T> for T {
    fn len(&self) -> usize {
        1
    }

    fn get(&self, i: usize) -> T {
        assert!(i == 0, "Index out of bounds");
        *self
    }

    fn set(&mut self, i: usize, value: T) {
        assert!(i == 0, "Index out of bounds");
        *self = value;
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
        assert!(i < 2, "Index out of bounds");
        if i == 0 { self.re } else { self.im }
    }

    fn set(&mut self, i: usize, value: T) {
        assert!(i < 2, "Index out of bounds");
        if i == 0 {
            self.re = value;
        } else {
            self.im = value;
        }
    }

    fn zeros() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}


/// A zero-sized state type for use when a feature (like Quadrature) is disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmptyState;

impl<T: Real> State<T> for EmptyState {
    fn len(&self) -> usize { 0 }
    fn get(&self, _i: usize) -> T { panic!("Index out of bounds") }
    fn set(&mut self, _i: usize, _value: T) { panic!("Index out of bounds") }
    fn zeros() -> Self { EmptyState }
}

impl Add for EmptyState { type Output = EmptyState; fn add(self, _: EmptyState) -> EmptyState { EmptyState } }
impl Sub for EmptyState { type Output = EmptyState; fn sub(self, _: EmptyState) -> EmptyState { EmptyState } }
impl AddAssign for EmptyState { fn add_assign(&mut self, _: EmptyState) {} }
impl<T: Real> Mul<T> for EmptyState { type Output = EmptyState; fn mul(self, _: T) -> EmptyState { EmptyState } }
impl<T: Real> Div<T> for EmptyState { type Output = EmptyState; fn div(self, _: T) -> EmptyState { EmptyState } }
impl Neg for EmptyState { type Output = EmptyState; fn neg(self) -> EmptyState { EmptyState } }

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_state_f64_get_out_of_bounds() {
        let state: f64 = 1.0;
        let _ = state.get(1);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_state_f64_set_out_of_bounds() {
        let mut state: f64 = 1.0;
        state.set(1, 2.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_state_complex_get_out_of_bounds() {
        let state = Complex::new(1.0, 2.0);
        let _ = state.get(2);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_state_complex_set_out_of_bounds() {
        let mut state = Complex::new(1.0, 2.0);
        state.set(2, 3.0);
    }
}
