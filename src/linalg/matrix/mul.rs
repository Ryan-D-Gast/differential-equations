//! Multiplication for Matrix: scalar, vector (State), and matrix product.

use core::ops::{Mul, MulAssign};

use crate::traits::{Real, State};

use super::base::{Matrix, MatrixStorage};

// Matrix * scalar (elementwise scale)
impl<T> Mul<T> for Matrix<T>
where
    T: Real,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        match self.storage {
            MatrixStorage::Identity => Matrix::diagonal(vec![rhs; self.nrows]),
            MatrixStorage::Full => Matrix { nrows: self.nrows, ncols: self.ncols, data: self.data.into_iter().map(|x| x * rhs).collect(), storage: MatrixStorage::Full },
            MatrixStorage::Banded { ml, mu, zero: _ } => Matrix { nrows: self.nrows, ncols: self.ncols, data: self.data.into_iter().map(|x| x * rhs).collect(), storage: MatrixStorage::Banded { ml, mu, zero: T::zero() } },
        }
    }
}

// In-place scalar scale: self *= scalar
impl<T> MulAssign<T> for Matrix<T>
where
    T: Real,
{
    fn mul_assign(&mut self, rhs: T) {
    let n = self.n();
    let lhs = core::mem::replace(self, Matrix::zeros(n));
        *self = lhs * rhs;
    }
}

// Matrix * State (vector-like) multiplication
impl<T: Real> Matrix<T> {
    pub fn mul_state<V: State<T>>(&self, vec: &V) -> V {
        let n = self.n();
        assert_eq!(
            vec.len(),
            n,
            "dimension mismatch in Matrix::mul_state"
        );

        let mut result = V::zeros();
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + self[(i, j)] * vec.get(j);
            }
            result.set(i, sum);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn mul_matrix_full() {
        let a: Matrix<f64> = Matrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = 5.0;
        let out = a * s;
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 10.0);
        assert_eq!(out[(1, 0)], 15.0);
        assert_eq!(out[(1, 1)], 20.0);
    }

    #[test]
    fn mul_identity() {
        let a: Matrix<f64> = Matrix::identity(2);
        let s = 5.0;
        let out = a * s;
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 0.0);
        assert_eq!(out[(1, 0)], 0.0);
        assert_eq!(out[(1, 1)], 5.0);
    }

    #[test]
    fn mul_assign() {
        let mut a: Matrix<f64> = Matrix::identity(2);
        let s = 5.0;
        a *= s;
        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], 0.0);
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(1, 1)], 5.0);
    }
}
