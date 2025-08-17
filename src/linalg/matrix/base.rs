//! Core matrix type, storage enum, and constructors.

use crate::traits::Real;

/// Matrix storage layout.
#[derive(Clone, Debug)]
pub enum MatrixStorage<T: Real> {
    /// Identity matrix (implicit). `data` stores [one, zero] to satisfy indexing by reference.
    Identity,
    /// Dense row-major matrix (nrows*ncols entries).
    Full,
    /// Banded matrix with lower (ml) and upper (mu) bandwidth.
    /// Compact diagonal storage with shape (ml+mu+1, ncols), row-major per diagonal.
    /// Off-band reads return `zero`.
    Banded { ml: usize, mu: usize, zero: T },
}

/// Generic matrix for linear algebra (typically square in current use).
#[derive(Clone, Debug)]
pub struct Matrix<T: Real> {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<T>,
    pub storage: MatrixStorage<T>,
}

impl<T: Real> Matrix<T> {
    /// Identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        Matrix {
            nrows: n,
            ncols: n,
            // Keep [one, zero] so indexing can return references.
            data: vec![T::one(), T::zero()],
            storage: MatrixStorage::Identity,
        }
    }

    /// Full matrix from a row-major vector of length n*n.
    pub fn full(n: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), n * n, "Matrix::full expects data of length n*n");
        Matrix {
            nrows: n,
            ncols: n,
            data,
            storage: MatrixStorage::Full,
        }
    }

    /// Zero matrix of size n x n.
    pub fn zeros(n: usize) -> Self {
        Matrix {
            nrows: n,
            ncols: n,
            data: vec![T::zero(); n * n],
            storage: MatrixStorage::Full,
        }
    }

    /// Zero banded matrix with the given bandwidths.
    /// For entry (i,j) within the band, index maps to data[i - j + mu, j].
    pub fn banded(n: usize, ml: usize, mu: usize) -> Self {
        let rows = ml + mu + 1;
        let data = vec![T::zero(); rows * n];
        Matrix {
            nrows: n,
            ncols: n,
            data,
            storage: MatrixStorage::Banded {
                ml,
                mu,
                zero: T::zero(),
            },
        }
    }

    /// Diagonal matrix from the provided diagonal entries (ml=mu=0).
    pub fn diagonal(diag: Vec<T>) -> Self {
        let n = diag.len();
        // With ml=mu=0, storage is (1,n), so `diag` maps directly to row 0.
        Matrix {
            nrows: n,
            ncols: n,
            data: diag,
            storage: MatrixStorage::Banded {
                ml: 0,
                mu: 0,
                zero: T::zero(),
            },
        }
    }

    /// Zero lower-triangular matrix (ml = n-1, mu = 0).
    pub fn lower_triangular(n: usize) -> Self {
        Matrix::banded(n, n.saturating_sub(1), 0)
    }

    /// Zero upper-triangular matrix (ml = 0, mu = n-1).
    pub fn upper_triangular(n: usize) -> Self {
        Matrix::banded(n, 0, n.saturating_sub(1))
    }

    /// Dimensions (nrows, ncols).
    pub fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Convenience: n for an n x n matrix.
    pub fn n(&self) -> usize {
        self.dims().0
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn diagonal_constructor_sets_diagonal() {
        let m = Matrix::diagonal(vec![1.0f64, 2.0, 3.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(2, 0)], 0.0);
    }

    #[test]
    fn triangular_constructors_shape() {
        let l: Matrix<f64> = Matrix::lower_triangular(4);
        // Above main diagonal reads zero
        assert_eq!(l[(0, 3)], 0.0);
        let u: Matrix<f64> = Matrix::upper_triangular(4);
        // Below main diagonal reads zero
        assert_eq!(u[(3, 0)], 0.0);
    }
}
