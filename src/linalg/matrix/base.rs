//! Core Matrix struct, storage enum, and constructors.

use crate::traits::Real;

/// Storage format of a Matrix.
#[derive(Clone, Debug)]
pub enum MatrixStorage<T: Real> {
    /// Identity matrix (implicitly stored). Data holds [one, zero] for references.
    Identity,
    /// Full dense matrix (data holds all entries row-major).
    Full,
    /// Banded matrix with lower (ml) and upper (mu) bandwidth
    /// stored in diagonal-wise compact form with shape (ml+mu+1, ncols)
    /// Layout: data[row * ncols + col]. Includes a cached zero for OOB reads.
    Banded { ml: usize, mu: usize, zero: T },
}

/// Generic matrix representation used for mass/jacobian matrices (square by usage today).
#[derive(Clone, Debug)]
pub struct Matrix<T: Real> {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<T>,
    pub storage: MatrixStorage<T>,
}

impl<T: Real> Matrix<T> {
    /// Construct an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        Matrix {
            nrows: n,
            ncols: n,
            // Store [one, zero] so we can return references for reads
            data: vec![T::one(), T::zero()],
            storage: MatrixStorage::Identity,
        }
    }

    /// Construct a full matrix from a row-major vector of length n*n.
    pub fn full(n: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            n * n,
            "Matrix::full expects data of length n*n"
        );
        Matrix { nrows: n, ncols: n, data, storage: MatrixStorage::Full }
    }

    /// Construct a zero matrix of size n x n.
    pub fn zeros(n: usize) -> Self {
        Matrix { nrows: n, ncols: n, data: vec![T::zero(); n * n], storage: MatrixStorage::Full }
    }

    /// Construct an empty banded matrix (all zeros) with the given size and bandwidths.
    /// Storage matches Fortran/LAPACK: data has shape (ml + mu + 1, n), and
    /// an element at (i, j) maps to data[i - j + mu, j] when within band.
    pub fn banded(n: usize, ml: usize, mu: usize) -> Self {
        let rows = ml + mu + 1;
        let data = vec![T::zero(); rows * n];
        Matrix {
            nrows: n,
            ncols: n,
            data,
            storage: MatrixStorage::Banded { ml, mu, zero: T::zero() },
        }
    }

    /// Construct a diagonal matrix from a vector of diagonal entries.
    /// Uses banded storage with ml=0, mu=0.
    pub fn diagonal(diag: Vec<T>) -> Self {
        let n = diag.len();
        // For ml=mu=0, banded storage has shape (1, n) and maps the main diagonal
        // to row 0, so the layout matches `diag` exactly. Move `diag` directly.
        Matrix {
            nrows: n,
            ncols: n,
            data: diag,
            storage: MatrixStorage::Banded { ml: 0, mu: 0, zero: T::zero() },
        }
    }

    /// Construct a zero lower-triangular matrix (including main diagonal).
    /// Uses banded storage with ml = n-1, mu = 0.
    pub fn lower_triangular(n: usize) -> Self {
        Matrix::banded(n, n.saturating_sub(1), 0)
    }

    /// Construct a zero upper-triangular matrix (including main diagonal).
    /// Uses banded storage with ml = 0, mu = n-1.
    pub fn upper_triangular(n: usize) -> Self {
        Matrix::banded(n, 0, n.saturating_sub(1))
    }

    /// Matrix dimensions (nrows, ncols).
    pub fn dims(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Convenience: size n for a square n x n matrix.
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
