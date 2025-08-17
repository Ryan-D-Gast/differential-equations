//! Matrix multiplication helpers.

use crate::traits::{Real, State};

use super::base::{Matrix, MatrixStorage};

// Matrix * State (vector-like)
impl<T: Real> Matrix<T> {
    /// Return a new matrix where each stored entry is multiplied by `rhs`.
    pub fn component_mul(mut self, rhs: T) -> Self {
        match &mut self.storage {
            MatrixStorage::Identity => Matrix::diagonal(vec![rhs; self.nrows]),
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v = *v * rhs;
                }
                self
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let n = self.nrows;
                let data = self.data.into_iter().map(|x| x * rhs).collect();
                Matrix {
                    nrows: n,
                    ncols: n,
                    data,
                    storage: MatrixStorage::Banded {
                        ml: *ml,
                        mu: *mu,
                        zero: T::zero(),
                    },
                }
            }
        }
    }

    /// In-place component-wise scalar multiplication: self[i,j] *= rhs for all stored entries.
    /// For Identity, converts to a diagonal banded matrix with `rhs` on the diagonal.
    pub fn component_mul_mut(&mut self, rhs: T) {
        match &mut self.storage {
            MatrixStorage::Identity => {
                // Become diagonal with rhs on the main diagonal
                let n = self.nrows;
                self.data = vec![rhs; n];
                self.storage = MatrixStorage::Banded {
                    ml: 0,
                    mu: 0,
                    zero: T::zero(),
                };
            }
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v = *v * rhs;
                }
            }
            MatrixStorage::Banded { .. } => {
                for v in &mut self.data {
                    *v = *v * rhs;
                }
            }
        }
    }

    pub fn mul_state<V: State<T>>(&self, vec: &V) -> V {
        let n = self.n();
        assert_eq!(vec.len(), n, "dimension mismatch in Matrix::mul_state");

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
        let out = a.component_mul(s);
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 10.0);
        assert_eq!(out[(1, 0)], 15.0);
        assert_eq!(out[(1, 1)], 20.0);
    }

    #[test]
    fn mul_identity() {
        let a: Matrix<f64> = Matrix::identity(2);
        let s = 5.0;
        let out = a.component_mul(s);
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 0.0);
        assert_eq!(out[(1, 0)], 0.0);
        assert_eq!(out[(1, 1)], 5.0);
    }

    #[test]
    fn mul_assign() {
        let a: Matrix<f64> = Matrix::identity(2);
        let s = 5.0;
        let a = a.component_mul(s);
        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], 0.0);
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(1, 1)], 5.0);
    }
}
