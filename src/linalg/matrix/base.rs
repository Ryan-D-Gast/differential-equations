//! Core SquareMatrix enum and constructors.

use crate::traits::Real;

/// Generic matrix representation used for mass/jacobian matrices.
#[derive(Clone, Debug)]
pub enum SquareMatrix<T: Real> {
    /// Identity matrix of size n x n (implicitly stored)
    Identity { n: usize, zero: T, one: T },

    /// Full dense matrix
    Full { n: usize, data: Vec<T> },

    /// Banded matrix with lower (ml) and upper (mu) bandwidth
    /// stored in diagonal-wise compact form with shape (ml+mu+1, n)
    Banded {
        n: usize,
        ml: usize,
        mu: usize,
        /// Row-major data with (ml + mu + 1) rows and n columns
        /// Layout: data[row * n + col]
        data: Vec<T>,
        /// A cached zero value so we can return references for out-of-band reads
        zero: T,
    },
}

impl<T: Real> SquareMatrix<T> {
    /// Construct an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        SquareMatrix::Identity {
            n,
            zero: T::zero(),
            one: T::one(),
        }
    }

    /// Construct a full matrix from a row-major vector of length n*n.
    pub fn full(n: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            n * n,
            "SquareMatrix::full expects data of length n*n"
        );
        SquareMatrix::Full { n, data }
    }

    /// Construct a zero matrix of size n x n.
    pub fn zeros(n: usize) -> Self {
        SquareMatrix::Full {
            n,
            data: vec![T::zero(); n * n],
        }
    }

    /// Construct an empty banded matrix (all zeros) with the given size and bandwidths.
    /// Storage matches Fortran/LAPACK: data has shape (ml + mu + 1, n), and
    /// an element at (i, j) maps to data[i - j + mu, j] when within band.
    pub fn banded(n: usize, ml: usize, mu: usize) -> Self {
        let rows = ml + mu + 1;
        let data = vec![T::zero(); rows * n];
        SquareMatrix::Banded {
            n,
            ml,
            mu,
            data,
            zero: T::zero(),
        }
    }

    /// Construct a diagonal matrix from a vector of diagonal entries.
    /// Uses banded storage with ml=0, mu=0.
    pub fn diagonal(diag: Vec<T>) -> Self {
        let n = diag.len();
        let mut m = SquareMatrix::banded(n, 0, 0);
        if let SquareMatrix::Banded { data, .. } = &mut m {
            for j in 0..n {
                data[0 * n + j] = diag[j].clone();
            }
        }
        m
    }

    /// Construct a zero lower-triangular matrix (including main diagonal).
    /// Uses banded storage with ml = n-1, mu = 0.
    pub fn lower_triangular(n: usize) -> Self {
        SquareMatrix::banded(n, n.saturating_sub(1), 0)
    }

    /// Construct a zero upper-triangular matrix (including main diagonal).
    /// Uses banded storage with ml = 0, mu = n-1.
    pub fn upper_triangular(n: usize) -> Self {
        SquareMatrix::banded(n, 0, n.saturating_sub(1))
    }

    /// SquareMatrix dimension (nrows, ncols). Always square for this enum.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            SquareMatrix::Identity { n, .. } => (*n, *n),
            SquareMatrix::Full { n, .. } => (*n, *n),
            SquareMatrix::Banded { n, .. } => (*n, *n),
        }
    }

    /// Size n for a square n x n matrix.
    pub fn n(&self) -> usize {
        self.dims().0
    }
}

#[cfg(test)]
mod tests {
    use super::SquareMatrix;

    #[test]
    fn diagonal_constructor_sets_diagonal() {
        let m = SquareMatrix::diagonal(vec![1.0f64, 2.0, 3.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(2, 0)], 0.0);
    }

    #[test]
    fn triangular_constructors_shape() {
        let l: SquareMatrix<f64> = SquareMatrix::lower_triangular(4);
        // Above main diagonal reads zero
        assert_eq!(l[(0, 3)], 0.0);
        let u: SquareMatrix<f64> = SquareMatrix::upper_triangular(4);
        // Below main diagonal reads zero
        assert_eq!(u[(3, 0)], 0.0);
    }
}
