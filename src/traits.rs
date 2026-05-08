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
/// * `SMatrix` - Matrix type from nalgebra
/// * `Complex` - Complex number type from num-complex
/// * `Struct<T>` - Any struct with all fields of type T using #[derive(State)] from the `derive` module
///
pub trait State<T: Real>: Clone + Debug {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, i: usize) -> T;

    fn set(&mut self, i: usize, value: T);

    /// Writes this state into a flat slice using the state's canonical solver layout.
    ///
    /// Backends with contiguous storage can override this to use `copy_from_slice`.
    fn write_to_slice(&self, output: &mut [T]) {
        assert_eq!(output.len(), self.len(), "Slice length mismatch");
        for (i, out) in output.iter_mut().enumerate() {
            *out = self.get(i);
        }
    }

    /// Updates this state from a flat slice using the state's canonical solver layout.
    ///
    /// This is the inverse of [`State::write_to_slice`] and is used to recover
    /// backend-specific states from solver-owned flat buffers.
    fn read_from_slice(&mut self, input: &[T]) {
        assert_eq!(input.len(), self.len(), "Slice length mismatch");
        for (i, value) in input.iter().copied().enumerate() {
            self.set(i, value);
        }
    }

    /// Constructs a zero-valued state with the same shape as `self`.
    fn zeros_like(&self) -> Self {
        Self::zeros()
    }

    fn zeros() -> Self;
}

pub trait StateAlgebra<T: Real>:
    State<T>
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + Neg<Output = Self>
{
}

impl<T, Y> StateAlgebra<T> for Y
where
    T: Real,
    Y: State<T>
        + Copy
        + Add<Output = Y>
        + Sub<Output = Y>
        + AddAssign
        + Mul<T, Output = Y>
        + Div<T, Output = Y>
        + Neg<Output = Y>,
{
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

    fn write_to_slice(&self, output: &mut [T]) {
        assert_eq!(output.len(), self.len(), "Slice length mismatch");
        for r in 0..R {
            for c in 0..C {
                output[r * C + c] = self[(r, c)];
            }
        }
    }

    fn read_from_slice(&mut self, input: &[T]) {
        assert_eq!(input.len(), self.len(), "Slice length mismatch");
        for r in 0..R {
            for c in 0..C {
                self[(r, c)] = input[r * C + c];
            }
        }
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

impl<T> State<T> for Vec<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, i: usize) -> T {
        self[i]
    }

    fn set(&mut self, i: usize, value: T) {
        self[i] = value;
    }

    fn write_to_slice(&self, output: &mut [T]) {
        output.copy_from_slice(self);
    }

    fn read_from_slice(&mut self, input: &[T]) {
        self.clear();
        self.extend_from_slice(input);
    }

    fn zeros_like(&self) -> Self {
        vec![T::zero(); self.len()]
    }

    fn zeros() -> Self {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

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

    #[test]
    fn test_smatrix_flat_slice_round_trip_uses_row_major_layout() {
        let state = SMatrix::<f64, 2, 2>::new(1.0, 2.0, 3.0, 4.0);
        let mut buffer = [0.0; 4];

        state.write_to_slice(&mut buffer);
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);

        let mut recovered = SMatrix::<f64, 2, 2>::zeros();
        recovered.read_from_slice(&buffer);
        assert_eq!(recovered, state);
    }

    #[test]
    fn test_complex_flat_slice_round_trip() {
        let state = Complex::new(1.0, 2.0);
        let mut buffer = [0.0; 2];

        state.write_to_slice(&mut buffer);
        assert_eq!(buffer, [1.0, 2.0]);

        let mut recovered = Complex::new(0.0, 0.0);
        recovered.read_from_slice(&buffer);
        assert_eq!(recovered, state);
    }
}
