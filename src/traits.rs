//! Defines Generics for the library. Includes generics for the floating point numbers.

#[cfg(feature = "nalgebra")]
use nalgebra::{DefaultAllocator, Dim, OMatrix, allocator::Allocator};
use num_complex::Complex;
use simba::scalar::{RealField, SupersetOf};
use std::fmt::Debug;

use crate::tolerance::Tolerance;

/// Real Number Trait
///
/// This trait specifies the acceptable types for real numbers.
/// Currently implemented for:
/// * `f32` - 32-bit floating point
/// * `f64` - 64-bit floating point
///
/// Provides additional functionality required for ODE solvers beyond
/// what's provided by simba's RealField trait.
///
pub trait Real: Copy + RealField + SupersetOf<f64> {
    fn infinity() -> Self;
}

impl<T: Copy + RealField + SupersetOf<f64>> Real for T {
    #[inline]
    fn infinity() -> Self {
        Self::from_subset(&f64::INFINITY)
    }
}

pub type DefaultState<T> = [T; 1];

/// State vector trait
///
/// Represents the state of the system being solved.
///
/// Implements for the following types:
/// * `[T; N]` - Fixed-size array state
/// * `OMatrix` - Matrix type from nalgebra, enabled with the `nalgebra` feature
/// * `Complex` - Complex number type from num-complex
/// * `Vec<T>` - Dynamically sized vector state
/// * `ndarray::Array<T, D>` - Array state, enabled with the `ndarray` feature
/// * `faer::Mat<T>` - Matrix state, enabled with the `faer` feature
/// * `Struct<T>` - Any struct with all fields of type T using #[derive(State)] from the `derive` module
///
pub trait State<T: Real>: Clone + Debug {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves a specific scalar component from the state using its flattened 1D index.
    fn get_component(&self, index: usize) -> T;

    /// Modifies a specific scalar component in the state using its flattened 1D index.
    fn set_component(&mut self, index: usize, value: T);

    /// Computes the element-wise absolute value, returning a new state.
    fn abs(&self) -> Self {
        let mut out = self.clone();
        out.map_components_mut(|_, val| *val = val.abs());
        out
    }

    /// Computes the element-wise product of two states.
    fn component_mul(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.map_components_mut(|i, val| *val *= other.get_component(i));
        out
    }

    /// Computes the element-wise division of two states.
    fn component_div(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.map_components_mut(|i, val| *val /= other.get_component(i));
        out
    }

    /// Computes the inner (dot) product between two states.
    fn dot(&self, other: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..self.len() {
            sum += self.get_component(i) * other.get_component(i);
        }
        sum
    }

    /// Computes the infinity norm (maximum absolute value) of the state.
    fn max_norm(&self) -> T {
        let mut max = T::zero();
        for i in 0..self.len() {
            max = max.max(self.get_component(i).abs());
        }
        max
    }

    /// Maps a closure over each component in the state alongside its flattened index.
    fn map_components_mut<F>(&mut self, f: F)
    where
        F: FnMut(usize, &mut T);

    /// In-place solution of a real linear system `LU * x = self`, where `LU` is a previously
    /// factorized dense matrix (in row-major order) and `ip` are the pivot indices.
    fn apply_linear_solve(&mut self, lu: &[T], ip: &[usize]) {
        let n = self.len();
        // Handle trivial case
        if n == 1 {
            self.set_component(0, self.get_component(0) / lu[0]);
            return;
        }

        let nm1 = n - 1;

        // Forward elimination with partial pivoting (solving Ly = Pb)
        for k in 0..nm1 {
            let kp1 = k + 1;
            let m = ip[k]; // Pivot row index

            // Apply row permutation to RHS
            let tk = self.get_component(k);
            let tm = self.get_component(m);
            self.set_component(k, tm);
            self.set_component(m, tk);

            // Forward substitution step
            let pivot_val = self.get_component(k);
            for i in kp1..n {
                let li_k = lu[i * n + k];
                let current = self.get_component(i);
                self.set_component(i, current + li_k * pivot_val);
            }
        }

        // Back substitution (solving Ux = y)
        for kb in 1..n {
            let k = n - kb;
            let diag = lu[k * n + k];
            let xk = self.get_component(k) / diag;
            self.set_component(k, xk);

            let neg_xk = -xk;
            for i in 0..k {
                let ui_k = lu[i * n + k];
                let current = self.get_component(i);
                self.set_component(i, current + ui_k * neg_xk);
            }
        }

        // Final division for the first element
        self.set_component(0, self.get_component(0) / lu[0]);
    }

    /// In-place solution of a complex linear system `(AR + i*AI) * (xr + i*xi) = self_r + i*self_i`.
    /// Updates both the real part (`self`) and the imaginary part (`imag_part`) in place.
    fn apply_complex_linear_solve(
        &mut self,
        imag_part: &mut Self,
        ar: &[T],
        ai: &[T],
        ip: &[usize],
    ) {
        let n = self.len();
        assert_eq!(imag_part.len(), n, "Complex linear solve dimension mismatch");

        // Handle trivial case
        if n == 1 {
            let br = self.get_component(0);
            let bi = imag_part.get_component(0);
            let ar_00 = ar[0];
            let ai_00 = ai[0];

            let denom = ar_00 * ar_00 + ai_00 * ai_00;
            self.set_component(0, (br * ar_00 + bi * ai_00) / denom);
            imag_part.set_component(0, (bi * ar_00 - br * ai_00) / denom);
            return;
        }

        let nm1 = n - 1;

        // Forward elimination with partial pivoting (solving complex Ly = Pb)
        for k in 0..nm1 {
            let kp1 = k + 1;
            let m = ip[k]; // Pivot row index

            // Apply row permutation to RHS
            let br_k = self.get_component(k);
            let bi_k = imag_part.get_component(k);
            let br_m = self.get_component(m);
            let bi_m = imag_part.get_component(m);

            self.set_component(k, br_m);
            imag_part.set_component(k, bi_m);
            self.set_component(m, br_k);
            imag_part.set_component(m, bi_k);

            // Forward substitution step (complex)
            let vr = self.get_component(k);
            let vi = imag_part.get_component(k);

            for i in kp1..n {
                let lr = ar[i * n + k];
                let li = ai[i * n + k];

                let current_r = self.get_component(i);
                let current_i = imag_part.get_component(i);

                // rhs[i] += L[i,k] * rhs[k]
                self.set_component(i, current_r + lr * vr - li * vi);
                imag_part.set_component(i, current_i + lr * vi + li * vr);
            }
        }

        // Back substitution (solving complex Ux = y)
        for kb in 1..n {
            let k = n - kb;

            // Complex division: rhs[k] /= U[k,k]
            let ur = ar[k * n + k];
            let ui = ai[k * n + k];
            let vr = self.get_component(k);
            let vi = imag_part.get_component(k);

            let denom = ur * ur + ui * ui;
            let xr = (vr * ur + vi * ui) / denom;
            let xi = (vi * ur - vr * ui) / denom;

            self.set_component(k, xr);
            imag_part.set_component(k, xi);

            // rhs[i] -= U[i,k] * x[k]
            let nxr = -xr;
            let nxi = -xi;

            for i in 0..k {
                let ur_ik = ar[i * n + k];
                let ui_ik = ai[i * n + k];

                let current_r = self.get_component(i);
                let current_i = imag_part.get_component(i);

                self.set_component(i, current_r + ur_ik * nxr - ui_ik * nxi);
                imag_part.set_component(i, current_i + ur_ik * nxi + ui_ik * nxr);
            }
        }

        // Final division for the first element (complex)
        let ur = ar[0];
        let ui = ai[0];
        let vr = self.get_component(0);
        let vi = imag_part.get_component(0);

        let denom = ur * ur + ui * ui;
        self.set_component(0, (vr * ur + vi * ui) / denom);
        imag_part.set_component(0, (vi * ur - vr * ui) / denom);
    }

    /// Multiplies the state by a dense matrix (in row-major order), returning a new state `y = A * x`.
    fn mul_by_dense_matrix(&self, matrix: &[T], n: usize, m: usize) -> Self {
        assert_eq!(self.len(), m, "Matrix-vector dimension mismatch");
        let mut out = self.zeros_like(); // Note: this assumes out should have same shape as self if it were square, but really it should have length n.
                                         // For ODEs n == m usually. If n != m, this might need a different constructor.
        if out.len() != n {
            // If the output state type is different (e.g. Vec), we might need to handle resizing.
            // For now assume n == m for simplicity as that is the common case in this library.
            assert_eq!(n, m, "mul_by_dense_matrix currently only supports square operations for generic States");
        }

        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..m {
                sum += matrix[i * m + j] * self.get_component(j);
            }
            out.set_component(i, sum);
        }
        out
    }

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
        self.map_components_mut(|_, val| *val = value);
    }

    /// Copy values from another state with the same flat solver layout.
    fn copy_from_state(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        self.map_components_mut(|i, val| *val = other.get_component(i));
    }

    /// In-place multiply and add with chaining: `self = self + alpha * other`.
    fn add_scaled(&mut self, alpha: T, other: &Self) -> &mut Self {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        self.map_components_mut(|i, val| *val += alpha * other.get_component(i));
        self
    }

    /// In-place scaling with chaining: `self = self * alpha`.
    fn scale_by(&mut self, alpha: T) -> &mut Self {
        self.map_components_mut(|_, val| *val *= alpha);
        self
    }

    /// Returns `self * alpha`.
    fn scaled(&self, alpha: T) -> Self {
        let mut out = self.clone();
        out.scale_by(alpha);
        out
    }

    /// Returns `self + alpha * other`.
    fn plus_scaled(&self, alpha: T, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_scaled(alpha, other);
        out
    }

    /// Returns `self` plus a linear combination of states.
    fn plus_linear_combination(&self, terms: &[(&Self, T)]) -> Self {
        let mut out = self.clone();
        for (state, alpha) in terms {
            out.add_scaled(*alpha, state);
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
            self.add_scaled(*alpha, state);
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
        let mut sum = T::zero();
        for i in 0..self.len() {
            let x = self.get_component(i);
            sum += x * x;
        }
        sum
    }

    /// Compute ||self - other||^2
    fn diff_norm_squared(&self, other: &Self) -> T {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        let mut sum = T::zero();
        for i in 0..self.len() {
            let diff = self.get_component(i) - other.get_component(i);
            sum += diff * diff;
        }
        sum
    }

    /// Calculates the weighted error norm used for adaptive step sizing.
    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.len(), y_new.len(), "State length mismatch");
        assert_eq!(self.len(), err.len(), "State length mismatch");

        let mut sum = T::zero();
        for i in 0..self.len() {
            let sk = atol[i] + rtol[i] * self.get_component(i).abs().max(y_new.get_component(i).abs());
            let e = err.get_component(i) / sk;
            sum += e * e;
        }
        sum
    }

    /// Calculates the maximum weighted error used by some adaptive step-size controllers.
    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.len(), y_new.len(), "State length mismatch");
        assert_eq!(self.len(), err.len(), "State length mismatch");

        let mut max = T::zero();
        for i in 0..self.len() {
            let sk = atol[i] + rtol[i] * self.get_component(i).abs().max(y_new.get_component(i).abs());
            max = max.max((err.get_component(i) / sk).abs());
        }
        max
    }
}

impl<T, const N: usize> State<T> for [T; N]
where
    T: Real,
{
    fn len(&self) -> usize {
        N
    }

    fn get_component(&self, index: usize) -> T {
        self[index]
    }

    fn set_component(&mut self, index: usize, value: T) {
        self[index] = value;
    }

    fn abs(&self) -> Self {
        let mut out = *self;
        for val in out.iter_mut() {
            *val = val.abs();
        }
        out
    }

    fn component_mul(&self, other: &Self) -> Self {
        let mut out = *self;
        for (v, o) in out.iter_mut().zip(other.iter()) {
            *v *= *o;
        }
        out
    }

    fn component_div(&self, other: &Self) -> Self {
        let mut out = *self;
        for (v, o) in out.iter_mut().zip(other.iter()) {
            *v /= *o;
        }
        out
    }

    fn dot(&self, other: &Self) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| sum + *a * *b)
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        for (i, val) in self.iter_mut().enumerate() {
            f(i, val);
        }
    }

    fn fill(&mut self, value: T) {
        self.as_mut_slice().fill(value);
    }

    fn copy_from_state(&mut self, other: &Self) {
        self.as_mut_slice().copy_from_slice(other);
    }

    fn norm_squared(&self) -> T {
        self.iter().fold(T::zero(), |sum, x| sum + *x * *x)
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| {
                let diff = *a - *b;
                sum + diff * diff
            })
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            let sk = atol[i] + rtol[i] * self[i].abs().max(y_new[i].abs());
            let e = err[i] / sk;
            sum += e * e;
        }
        sum
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        let mut max = T::zero();
        for i in 0..N {
            let sk = atol[i] + rtol[i] * self[i].abs().max(y_new[i].abs());
            max = max.max((err[i] / sk).abs());
        }
        max
    }

    fn zeros_like(&self) -> Self {
        [T::zero(); N]
    }

    fn zeros() -> Self {
        [T::zero(); N]
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

#[cfg(feature = "nalgebra")]
impl<T, R, C> State<T> for OMatrix<T, R, C>
where
    T: Real,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    fn get_component(&self, index: usize) -> T {
        let c = index % self.ncols();
        let r = index / self.ncols();
        self[(r, c)]
    }

    fn set_component(&mut self, index: usize, value: T) {
        let c = index % self.ncols();
        let r = index / self.ncols();
        self[(r, c)] = value;
    }

    fn abs(&self) -> Self {
        self.abs()
    }

    fn component_mul(&self, other: &Self) -> Self {
        self.component_mul(other)
    }

    fn component_div(&self, other: &Self) -> Self {
        self.component_div(other)
    }

    fn dot(&self, other: &Self) -> T {
        self.dot(other)
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        for i in 0..self.len() {
            let c = i % self.ncols();
            let r = i / self.ncols();
            f(i, &mut self[(r, c)]);
        }
    }

    fn apply_linear_solve(&mut self, lu: &[T], ip: &[usize]) {
        // Use default implementation for now as nalgebra integration of row-major LU is non-trivial.
        // In the future, we can add specialized paths.
        let n = self.len();
        if n == 1 {
            self.set_component(0, self.get_component(0) / lu[0]);
            return;
        }

        let nm1 = n - 1;
        for k in 0..nm1 {
            let m = ip[k];
            let tk = self.get_component(k);
            let tm = self.get_component(m);
            self.set_component(k, tm);
            self.set_component(m, tk);

            let pivot_val = self.get_component(k);
            for i in k + 1..n {
                let li_k = lu[i * n + k];
                let current = self.get_component(i);
                self.set_component(i, current + li_k * pivot_val);
            }
        }

        for kb in 1..n {
            let k = n - kb;
            let diag = lu[k * n + k];
            let xk = self.get_component(k) / diag;
            self.set_component(k, xk);

            let neg_xk = -xk;
            for i in 0..k {
                let ui_k = lu[i * n + k];
                let current = self.get_component(i);
                self.set_component(i, current + ui_k * neg_xk);
            }
        }

        self.set_component(0, self.get_component(0) / lu[0]);
    }

    fn zeros_like(&self) -> Self {
        let (nrows, ncols) = self.shape_generic();
        Self::zeros_generic(nrows, ncols)
    }

    fn zeros() -> Self {
        let nrows = R::from_usize(R::try_to_usize().unwrap_or(0));
        let ncols = C::from_usize(C::try_to_usize().unwrap_or(0));
        Self::zeros_generic(nrows, ncols)
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        *self += other * alpha;
    }

    fn scale_mut(&mut self, alpha: T) {
        *self *= alpha;
    }

    fn fill(&mut self, value: T) {
        self.fill(value);
    }

    fn copy_from_state(&mut self, other: &Self) {
        self.copy_from(other);
    }

    fn norm_squared(&self) -> T {
        self.norm_squared()
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        (self - other).norm_squared()
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.nrows(), y_new.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), y_new.ncols(), "State column count mismatch");
        assert_eq!(self.nrows(), err.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), err.ncols(), "State column count mismatch");
        let mut sum = T::zero();
        for i in 0..self.len() {
            let c = i % self.ncols();
            let r = i / self.ncols();
            let sk = atol[i] + rtol[i] * self[(r, c)].abs().max(y_new[(r, c)].abs());
            let e = err[(r, c)] / sk;
            sum += e * e;
        }
        sum
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.nrows(), y_new.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), y_new.ncols(), "State column count mismatch");
        assert_eq!(self.nrows(), err.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), err.ncols(), "State column count mismatch");
        let mut max = T::zero();
        for i in 0..self.len() {
            let c = i % self.ncols();
            let r = i / self.ncols();
            let sk = atol[i] + rtol[i] * self[(r, c)].abs().max(y_new[(r, c)].abs());
            max = max.max((err[(r, c)] / sk).abs());
        }
        max
    }
}

impl<T> State<T> for Complex<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        2
    }

    fn get_component(&self, index: usize) -> T {
        match index {
            0 => self.re,
            1 => self.im,
            _ => panic!("Index out of bounds for Complex state"),
        }
    }

    fn set_component(&mut self, index: usize, value: T) {
        match index {
            0 => self.re = value,
            1 => self.im = value,
            _ => panic!("Index out of bounds for Complex state"),
        }
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        f(0, &mut self.re);
        f(1, &mut self.im);
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

    fn fill(&mut self, value: T) {
        self.re = value;
        self.im = value;
    }

    fn copy_from_state(&mut self, other: &Self) {
        self.re = other.re;
        self.im = other.im;
    }

    fn norm_squared(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        let re = self.re - other.re;
        let im = self.im - other.im;
        re * re + im * im
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        let sk_re = atol[0] + rtol[0] * self.re.abs().max(y_new.re.abs());
        let sk_im = atol[1] + rtol[1] * self.im.abs().max(y_new.im.abs());
        let e_re = err.re / sk_re;
        let e_im = err.im / sk_im;
        e_re * e_re + e_im * e_im
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        let sk_re = atol[0] + rtol[0] * self.re.abs().max(y_new.re.abs());
        let sk_im = atol[1] + rtol[1] * self.im.abs().max(y_new.im.abs());
        (err.re / sk_re).abs().max((err.im / sk_im).abs())
    }
}

impl<T> State<T> for Vec<T>
where
    T: Real,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn get_component(&self, index: usize) -> T {
        self[index]
    }

    fn set_component(&mut self, index: usize, value: T) {
        self[index] = value;
    }

    fn dot(&self, other: &Self) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| sum + *a * *b)
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        for (i, val) in self.iter_mut().enumerate() {
            f(i, val);
        }
    }

    fn zeros_like(&self) -> Self {
        vec![T::zero(); self.len()]
    }

    fn zeros() -> Self {
        Vec::new()
    }

    fn mul_add_assign(&mut self, alpha: T, other: &Self) {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        for (s, o) in self.iter_mut().zip(other.iter()) {
            *s += alpha * *o;
        }
    }

    fn scale_mut(&mut self, alpha: T) {
        for s in self.iter_mut() {
            *s *= alpha;
        }
    }

    fn fill(&mut self, value: T) {
        self.as_mut_slice().fill(value);
    }

    fn copy_from_state(&mut self, other: &Self) {
        self.clone_from(other);
    }

    fn norm_squared(&self) -> T {
        self.iter().fold(T::zero(), |sum, x| sum + *x * *x)
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        assert_eq!(self.len(), other.len(), "State length mismatch");
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| {
                let diff = *a - *b;
                sum + diff * diff
            })
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.len(), y_new.len(), "State length mismatch");
        assert_eq!(self.len(), err.len(), "State length mismatch");
        let mut sum = T::zero();
        for i in 0..self.len() {
            let sk = atol[i] + rtol[i] * self[i].abs().max(y_new[i].abs());
            let e = err[i] / sk;
            sum += e * e;
        }
        sum
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.len(), y_new.len(), "State length mismatch");
        assert_eq!(self.len(), err.len(), "State length mismatch");
        let mut max = T::zero();
        for i in 0..self.len() {
            let sk = atol[i] + rtol[i] * self[i].abs().max(y_new[i].abs());
            max = max.max((err[i] / sk).abs());
        }
        max
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

    fn get_component(&self, index: usize) -> T {
        // ndarray::Array usually has a 1D iterator even for higher dimensional arrays
        *self.iter().nth(index).unwrap()
    }

    fn set_component(&mut self, index: usize, value: T) {
        *self.iter_mut().nth(index).unwrap() = value;
    }

    fn dot(&self, other: &Self) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| sum + *a * *b)
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        for (i, val) in self.iter_mut().enumerate() {
            f(i, val);
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

    fn fill(&mut self, value: T) {
        for dst in self.iter_mut() {
            *dst = value;
        }
    }

    fn copy_from_state(&mut self, other: &Self) {
        assert_eq!(self.shape(), other.shape(), "State shape mismatch");
        for (dst, src) in self.iter_mut().zip(other.iter()) {
            *dst = *src;
        }
    }

    fn norm_squared(&self) -> T {
        self.iter().fold(T::zero(), |sum, x| sum + *x * *x)
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        assert_eq!(self.shape(), other.shape(), "State shape mismatch");
        self.iter()
            .zip(other.iter())
            .fold(T::zero(), |sum, (a, b)| {
                let diff = *a - *b;
                sum + diff * diff
            })
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.shape(), y_new.shape(), "State shape mismatch");
        assert_eq!(self.shape(), err.shape(), "State shape mismatch");
        let mut sum = T::zero();
        for (i, ((y, y_new), err)) in self.iter().zip(y_new.iter()).zip(err.iter()).enumerate() {
            let sk = atol[i] + rtol[i] * (*y).abs().max((*y_new).abs());
            let e = *err / sk;
            sum += e * e;
        }
        sum
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.shape(), y_new.shape(), "State shape mismatch");
        assert_eq!(self.shape(), err.shape(), "State shape mismatch");
        let mut max = T::zero();
        for (i, ((y, y_new), err)) in self.iter().zip(y_new.iter()).zip(err.iter()).enumerate() {
            let sk = atol[i] + rtol[i] * (*y).abs().max((*y_new).abs());
            max = max.max((*err / sk).abs());
        }
        max
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

    fn get_component(&self, index: usize) -> T {
        let c = index % self.ncols();
        let r = index / self.ncols();
        *self.get(r, c)
    }

    fn set_component(&mut self, index: usize, value: T) {
        let c = index % self.ncols();
        let r = index / self.ncols();
        *self.get_mut(r, c) = value;
    }

    fn map_components_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut T),
    {
        for i in 0..self.len() {
            let c = i % self.ncols();
            let r = i / self.ncols();
            f(i, self.get_mut(r, c));
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

    fn fill(&mut self, value: T) {
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                *self.get_mut(r, c) = value;
            }
        }
    }

    fn copy_from_state(&mut self, other: &Self) {
        assert_eq!(self.nrows(), other.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), other.ncols(), "State column count mismatch");
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                *self.get_mut(r, c) = *other.get(r, c);
            }
        }
    }

    fn norm_squared(&self) -> T {
        let mut sum = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let value = *self.get(r, c);
                sum += value * value;
            }
        }
        sum
    }

    fn diff_norm_squared(&self, other: &Self) -> T {
        assert_eq!(self.nrows(), other.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), other.ncols(), "State column count mismatch");
        let mut sum = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let diff = *self.get(r, c) - *other.get(r, c);
                sum += diff * diff;
            }
        }
        sum
    }

    fn error_norm(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.nrows(), y_new.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), y_new.ncols(), "State column count mismatch");
        assert_eq!(self.nrows(), err.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), err.ncols(), "State column count mismatch");
        let mut sum = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let i = r * self.ncols() + c;
                let sk = atol[i] + rtol[i] * (*self.get(r, c)).abs().max((*y_new.get(r, c)).abs());
                let e = *err.get(r, c) / sk;
                sum += e * e;
            }
        }
        sum
    }

    fn error_norm_inf(
        &self,
        y_new: &Self,
        err: &Self,
        atol: &Tolerance<T>,
        rtol: &Tolerance<T>,
    ) -> T {
        assert_eq!(self.nrows(), y_new.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), y_new.ncols(), "State column count mismatch");
        assert_eq!(self.nrows(), err.nrows(), "State row count mismatch");
        assert_eq!(self.ncols(), err.ncols(), "State column count mismatch");
        let mut max = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let i = r * self.ncols() + c;
                let sk = atol[i] + rtol[i] * (*self.get(r, c)).abs().max((*y_new.get(r, c)).abs());
                max = max.max((*err.get(r, c) / sk).abs());
            }
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_array_get_set() {
        let mut state = [1.0, 2.0, 3.0];
        assert_eq!(state.get_component(0), 1.0);
        state.set_component(1, 42.0);
        assert_eq!(state[1], 42.0);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_nalgebra_matrix_get_set() {
        let mut state = nalgebra::SMatrix::<f64, 2, 2>::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(state.get_component(1), 2.0); // Row-major index 1 is (0, 1)
        state.set_component(2, 42.0); // Row-major index 2 is (1, 0)
        assert_eq!(state[(1, 0)], 42.0);
    }

    #[test]
    fn test_complex_get_set() {
        let mut state = Complex::new(1.0, 2.0);
        assert_eq!(state.get_component(0), 1.0);
        state.set_component(1, 42.0);
        assert_eq!(state.im, 42.0);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_get_set() {
        let mut state = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(state.get_component(1), 2.0);
        state.set_component(2, 42.0);
        assert_eq!(state[[1, 0]], 42.0);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn test_faer_get_set() {
        let mut state = faer::Mat::from_fn(2, 2, |r, c| (r * 2 + c + 1) as f64);
        assert_eq!(state.get_component(1), 2.0);
        state.set_component(2, 42.0);
        assert_eq!(*state.get(1, 0), 42.0);
    }
}
