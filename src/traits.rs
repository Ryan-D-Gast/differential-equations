//! Defines Generics for the library. Includes generics for the floating point numbers.

use nalgebra::{RealField, SMatrix};
use num_complex::Complex;
use std::fmt::Debug;

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

    /// Writes this state into a flat slice using the state's canonical solver layout.
    ///
    /// Backends with contiguous storage can override this to use `copy_from_slice`.
    fn write_to_slice(&self, output: &mut [T]);

    /// Updates this state from a flat slice using the state's canonical solver layout.
    ///
    /// This is the inverse of [`State::write_to_slice`] and is used to recover
    /// backend-specific states from solver-owned flat buffers.
    fn read_from_slice(&mut self, input: &[T]);

    /// Constructs a zero-valued state with the same shape as `self`.
    fn zeros_like(&self) -> Self;

    /// Constructs a default zero-valued state.
    ///
    /// Dynamically sized backends should return an empty state here and use
    /// [`State::zeros_like`] once an initial condition provides the runtime shape.
    fn zeros() -> Self;

    /// In-place multiply and add: `self = self + alpha * other`
    fn mul_add_assign(&mut self, alpha: T, other: &Self);

    /// In-place scaling: `self = self * alpha`
    fn scale_mut(&mut self, alpha: T);

    /// Fill with a constant value
    fn fill(&mut self, value: T) {
        let mut buf = vec![T::zero(); self.len()];
        for x in buf.iter_mut() {
            *x = value;
        }
        self.read_from_slice(&buf);
    }

    /// Copy values from another state with the same flat solver layout.
    fn copy_from_state(&mut self, other: &Self) {
        let mut buf = vec![T::zero(); other.len()];
        other.write_to_slice(&mut buf);
        self.read_from_slice(&buf);
    }

    /// In-place multiply and add with chaining: `self = self + alpha * other`.
    fn add_scaled(&mut self, alpha: T, other: &Self) -> &mut Self {
        self.mul_add_assign(alpha, other);
        self
    }

    /// In-place scaling with chaining: `self = self * alpha`.
    fn scale_by(&mut self, alpha: T) -> &mut Self {
        self.scale_mut(alpha);
        self
    }

    /// Returns `self * alpha`.
    fn scaled(&self, alpha: T) -> Self {
        let mut out = self.clone();
        out.scale_mut(alpha);
        out
    }

    /// Returns `self + alpha * other`.
    fn plus_scaled(&self, alpha: T, other: &Self) -> Self {
        let mut out = self.clone();
        out.mul_add_assign(alpha, other);
        out
    }

    /// Returns `self` plus a linear combination of states.
    fn plus_linear_combination(&self, terms: &[(&Self, T)]) -> Self {
        let mut out = self.clone();
        for (state, alpha) in terms {
            out.mul_add_assign(*alpha, state);
        }
        out
    }

    /// Returns `self - other`.
    fn minus(&self, other: &Self) -> Self {
        self.plus_scaled(-T::one(), other)
    }

    /// Sets `self` to a linear combination of states.
    fn set_linear_combination(&mut self, terms: &[(&Self, T)]) -> &mut Self {
        self.fill(T::zero());
        for (state, alpha) in terms {
            self.mul_add_assign(*alpha, state);
        }
        self
    }

    /// Returns a linear combination with the same shape as `self`.
    fn linear_combination(&self, terms: &[(&Self, T)]) -> Self {
        let mut out = self.zeros_like();
        out.set_linear_combination(terms);
        out
    }

    /// Compute ||self||^2
    fn norm_squared(&self) -> T {
        let mut buf = vec![T::zero(); self.len()];
        self.write_to_slice(&mut buf);
        let mut sum = T::zero();
        for &x in &buf {
            sum += x * x;
        }
        sum
    }

    /// Compute ||self - other||^2
    fn diff_norm_squared(&self, other: &Self) -> T {
        let mut buf_self = vec![T::zero(); self.len()];
        let mut buf_other = vec![T::zero(); self.len()];
        self.write_to_slice(&mut buf_self);
        other.write_to_slice(&mut buf_other);
        let mut sum = T::zero();
        for (a, b) in buf_self.iter().zip(buf_other.iter()) {
            let diff = *a - *b;
            sum += diff * diff;
        }
        sum
    }

    /// Calculates the weighted error norm used for adaptive step sizing.
    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &crate::tolerance::Tolerance<T>,
        rtol: &crate::tolerance::Tolerance<T>,
    ) -> T {
        let mut buf_y = vec![T::zero(); self.len()];
        let mut buf_y_new = vec![T::zero(); self.len()];
        let mut buf_err = vec![T::zero(); self.len()];
        self.write_to_slice(&mut buf_y);
        y_new.write_to_slice(&mut buf_y_new);
        err.write_to_slice(&mut buf_err);

        let mut sum = T::zero();
        for i in 0..self.len() {
            let sk = atol[i] + rtol[i] * buf_y[i].abs().max(buf_y_new[i].abs());
            let e = buf_err[i] / sk;
            sum += e * e;
        }
        sum
    }
}

impl<T, const R: usize, const C: usize> State<T> for SMatrix<T, R, C>
where
    T: Real,
{
    fn len(&self) -> usize {
        R * C
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

    fn zeros_like(&self) -> Self {
        SMatrix::<T, R, C>::zeros()
    }

    fn zeros() -> Self {
        SMatrix::<T, R, C>::zeros()
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        for i in 0..self.len() {
            self[(i / C, i % C)] += alpha * other[(i / C, i % C)];
        }
    }

    fn scale_mut(&mut self, alpha: T) {
        for i in 0..self.len() {
            self[(i / C, i % C)] *= alpha;
        }
    }
}

impl<T> State<T> for Complex<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        2
    }

    fn write_to_slice(&self, output: &mut [T]) {
        assert_eq!(output.len(), 2, "Slice length mismatch");
        output[0] = self.re;
        output[1] = self.im;
    }

    fn read_from_slice(&mut self, input: &[T]) {
        assert_eq!(input.len(), 2, "Slice length mismatch");
        self.re = input[0];
        self.im = input[1];
    }

    fn zeros_like(&self) -> Self {
        Complex::new(T::zero(), T::zero())
    }

    fn zeros() -> Self {
        Complex::new(T::zero(), T::zero())
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        self.re += alpha * other.re;
        self.im += alpha * other.im;
    }

    fn scale_mut(&mut self, alpha: T) {
        self.re *= alpha;
        self.im *= alpha;
    }
}

impl<T> State<T> for Vec<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        self.len()
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

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        for (s, o) in self.iter_mut().zip(other.iter()) {
            *s += alpha * *o;
        }
    }

    fn scale_mut(&mut self, alpha: T) {
        for s in self.iter_mut() {
            *s *= alpha;
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T, D> State<T> for ndarray::Array<T, D>
where
    T: Real,
    D: ndarray::Dimension,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn write_to_slice(&self, output: &mut [T]) {
        assert_eq!(output.len(), self.len(), "Slice length mismatch");
        for (dst, src) in output.iter_mut().zip(self.iter()) {
            *dst = *src;
        }
    }

    fn read_from_slice(&mut self, input: &[T]) {
        assert_eq!(input.len(), self.len(), "Slice length mismatch");
        for (dst, src) in self.iter_mut().zip(input.iter()) {
            *dst = *src;
        }
    }

    fn zeros_like(&self) -> Self {
        ndarray::Array::zeros(self.raw_dim())
    }

    fn zeros() -> Self {
        ndarray::Array::zeros(D::default())
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        for (dst, src) in self.iter_mut().zip(other.iter()) {
            *dst += alpha * *src;
        }
    }

    fn scale_mut(&mut self, alpha: T) {
        for dst in self.iter_mut() {
            *dst *= alpha;
        }
    }
}

#[cfg(feature = "faer")]
impl<T> State<T> for faer::Mat<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    fn write_to_slice(&self, output: &mut [T]) {
        assert_eq!(output.len(), self.len(), "Slice length mismatch");
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                output[r * self.ncols() + c] = *self.get(r, c);
            }
        }
    }

    fn read_from_slice(&mut self, input: &[T]) {
        assert_eq!(input.len(), self.len(), "Slice length mismatch");
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                *self.get_mut(r, c) = input[r * self.ncols() + c];
            }
        }
    }

    fn zeros_like(&self) -> Self {
        faer::Mat::from_fn(self.nrows(), self.ncols(), |_, _| T::zero())
    }

    fn zeros() -> Self {
        faer::Mat::from_fn(0, 0, |_, _| T::zero())
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        assert_eq!(self.nrows(), other.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), other.ncols(), "State column count mismatch");
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                *self.get_mut(r, c) += alpha * *other.get(r, c);
            }
        }
    }

    fn scale_mut(&mut self, alpha: T) {
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                *self.get_mut(r, c) *= alpha;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

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

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_flat_slice_round_trip() {
        let state = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let mut buffer = [0.0; 4];

        state.write_to_slice(&mut buffer);
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);

        let mut recovered = state.zeros_like();
        recovered.read_from_slice(&buffer);
        assert_eq!(recovered, state);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_faer_flat_slice_round_trip() {
        let state = faer::Mat::from_fn(2, 2, |r, c| (r * 2 + c + 1) as f64);
        let mut buffer = [0.0; 4];

        state.write_to_slice(&mut buffer);
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);

        let mut recovered = state.zeros_like();
        recovered.read_from_slice(&buffer);
        assert_eq!(*recovered.get(0, 0), 1.0);
        assert_eq!(*recovered.get(0, 1), 2.0);
        assert_eq!(*recovered.get(1, 0), 3.0);
        assert_eq!(*recovered.get(1, 1), 4.0);
    }
}
