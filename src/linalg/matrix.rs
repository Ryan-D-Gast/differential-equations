//! Matrix storage enum compatible with Fortran band/full conventions
//!
//! This provides an enum that can represent:
//! - Identity matrices (implicit storage)
//! - Full/dense matrices (row-major Vec<T>)
//! - Banded matrices using the same diagonal-wise storage as classic Fortran/LAPACK
//!   band storage (aka GB-matrix).
//!
//! Identity (n = 4) looks like:
//!
//! ```text
//! [1 0 0 0]
//! [0 1 0 0]
//! [0 0 1 0]
//! [0 0 0 1]
//! ```
//!
//! Full (row-major) stores the whole n×n matrix in a single Vec<T> of length n*n.
//! The entry M[i,j] is at index i*n + j.
//!
//! Banded stores only a band around the main diagonal. With lower bandwidth `ml`
//! and upper bandwidth `mu`, the nonzeros satisfy:
//!
//! ```text
//!    -ml <= i - j <= mu
//! ```
//!
//! The storage is a compact array with (ml + mu + 1) rows and n columns, laid out
//! row-major in a Vec<T> of length (ml + mu + 1) * n. The mapping is:
//!
//! ```text
//! storage_row = i - j + mu   // 0-based
//! data[storage_row * n + j] = M[i, j]
//! ```
//!
//! Example: For n = 5, ml = 1, mu = 2, M has the shape (x marks possible nonzeros):
//!
//! ```text
//! [x x x 0 0]
//! [x x x x 0]
//! [0 x x x x]
//! [0 0 x x x]
//! [0 0 0 x x]
//! ```
//!
//! Its compact band storage (rows are upper-2, upper-1, main, lower-1) is a 4×5 array:
//!
//! ```text
//! [u2_0 u2_1 u2_2 u2_3 u2_4]  // i - j = -2 (2nd upper diagonal)
//! [u1_0 u1_1 u1_2 u1_3 u1_4]  // i - j = -1 (1st upper diagonal)
//! [ d_0  d_1  d_2  d_3  d_4]  // i - j =  0 (main diagonal)
//! [ l1_0 l1_1 l1_2 l1_3 l1_4] // i - j = +1 (1st lower diagonal)
//! ```
//!
//! Entries that fall outside the band are implicitly zero and not stored.
//!
//! Mutability and automatic conversion:
//! - Writing to an Identity matrix converts it to a Full matrix on-the-fly.
//! - Writing outside the band of a Banded matrix converts it to a Full matrix on-the-fly.
//! This preserves intuitive behavior while keeping storage compact when possible.
//!
//! Example:
//! ```
//! use differential_equations::linalg::Matrix;
//! // Identity -> write off-diagonal -> becomes Full
//! let mut m: Matrix<f64> = Matrix::identity(3);
//! m[(2, 0)] = 5.0; // converts to Full internally
//! assert_eq!(m[(2,0)], 5.0);
//! // Banded (ml=0, mu=0) -> write outside band -> becomes Full
//! let mut b: Matrix<f64> = Matrix::banded(3, 0, 0); // only main diagonal
//! b[(0, 2)] = 7.0; // converts to Full internally
//! assert_eq!(b[(0,2)], 7.0);
//! ```

use crate::traits::Real;
use core::fmt::{self, Display, Write as _};
use core::ops::{Index, IndexMut};

/// Generic matrix representation used for mass/jacobian matrices.
#[derive(Clone, Debug)]
pub enum Matrix<T: Real> {
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

impl<T: Real> Matrix<T> {
    /// Construct an identity matrix of size n x n.
    ///
    /// # Examples
    /// ```
    /// use differential_equations::linalg::Matrix;
    /// let m: Matrix<f64> = Matrix::identity(3);
    /// match m {
    ///     Matrix::Identity { n, .. } => assert_eq!(n, 3),
    ///     _ => unreachable!(),
    /// }
    /// ```
    pub fn identity(n: usize) -> Self {
        Matrix::Identity {
            n,
            zero: T::zero(),
            one: T::one(),
        }
    }

    /// Construct a full matrix from a row-major vector of length n*n.
    ///
    /// The input `data` must be length `n*n` and is interpreted as row-major,
    /// so the entry at (i, j) is stored at `data[i*n + j]`.
    ///
    /// If `data = [1, 2, 3, 4]` and `n = 2`, the matrix is
    ///
    /// ```text
    /// [1 2]
    /// [3 4]
    /// ```
    ///
    /// # Examples
    /// ```
    /// use differential_equations::linalg::Matrix;
    /// let n = 2usize;
    /// let data = vec![1.0f64, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] row-major
    /// let m = Matrix::full(n, data);
    /// match m {
    ///     Matrix::Full { n, data } => {
    ///         assert_eq!(n, 2);
    ///         assert_eq!(data.len(), 4);
    ///     },
    ///     _ => unreachable!(),
    /// }
    /// // Index reads/writes
    /// let mut m = Matrix::full(2, vec![0.0, 0.0, 0.0, 0.0]);
    /// m[(0,0)] = 1.0;
    /// m[(0,1)] = 2.0;
    /// m[(1,0)] = 3.0;
    /// m[(1,1)] = 4.0;
    /// assert_eq!(m[(0,1)], 2.0);
    /// ```
    pub fn full(n: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), n * n, "Matrix::full expects data of length n*n");
        Matrix::Full { n, data }
    }

    /// Construct a zero matrix of size n x n.
    pub fn zeros(n: usize) -> Self {
        Matrix::Full {
            n,
            data: vec![T::zero(); n * n],
        }
    }

    /// Construct an empty banded matrix (all zeros) with the given size and bandwidths.
    /// Storage matches Fortran/LAPACK: data has shape (ml + mu + 1, n), and
    /// an element at (i, j) maps to data[i - j + mu, j] when within band.
    ///
    /// For example, with `n = 5`, `ml = 1`, `mu = 2`, the non-zero structure is:
    ///
    /// ```text
    /// [x x x 0 0]
    /// [x x x x 0]
    /// [0 x x x x]
    /// [0 0 x x x]
    /// [0 0 0 x x]
    /// ```
    ///
    /// The compact storage has `(ml + mu + 1) = 4` rows and 5 columns.
    /// Row indices map to diagonals as: `i - j = -2, -1, 0, +1`.
    ///
    /// # Examples
    /// ```
    /// use differential_equations::linalg::Matrix;
    /// // 5x5 matrix with ml=1, mu=2 -> storage has (1+2+1)=4 rows and 5 cols
    /// let m: Matrix<f64> = Matrix::banded(5, 1, 2);
    /// match m {
    ///     Matrix::Banded { n, ml, mu, data, .. } => {
    ///         assert_eq!(n, 5);
    ///         assert_eq!(ml, 1);
    ///         assert_eq!(mu, 2);
    ///         assert_eq!(data.len(), (ml + mu + 1) * n);
    ///     },
    ///     _ => unreachable!(),
    /// }
    /// // Writing within the band:
    /// let mut m = Matrix::banded(5, 1, 2);
    /// m[(2, 2)] = 10.0; // main diagonal
    /// m[(1, 2)] = 5.0;  // upper-1 diagonal (i-j = -1)
    /// m[(3, 2)] = 7.0;  // lower-1 diagonal (i-j = +1)
    /// assert_eq!(m[(2,2)], 10.0);
    /// assert_eq!(m[(1,2)], 5.0);
    /// assert_eq!(m[(3,2)], 7.0);
    /// // Reading outside the band returns 0 (and writing would panic):
    /// assert_eq!(m[(0,4)], 0.0);
    /// ```
    pub fn banded(n: usize, ml: usize, mu: usize) -> Self {
        let rows = ml + mu + 1;
        let data = vec![T::zero(); rows * n];
        Matrix::Banded {
            n,
            ml,
            mu,
            data,
            zero: T::zero(),
        }
    }

    /// Matrix dimension (nrows, ncols). Always square for this enum.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Matrix::Identity { n, .. } => (*n, *n),
            Matrix::Full { n, .. } => (*n, *n),
            Matrix::Banded { n, .. } => (*n, *n),
        }
    }

    /// Size n for a square n x n matrix.
    pub fn n(&self) -> usize {
        self.dims().0
    }

    /// Print a simple dense view of the matrix for debugging.
    pub fn debug_print(&self)
    where
        T: Display,
    {
        println!("{}", self);
    }
}

/// 2D indexing by (i, j): read-only.
///
/// Note:
/// - For Identity, off-diagonal reads return a reference to a cached zero.
/// - For Banded, out-of-band reads return a reference to a cached zero.
impl<T: Real> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        match self {
            Matrix::Identity { n, zero, one } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                if i == j { one } else { zero }
            }
            Matrix::Full { n, data } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                &data[i * n + j]
            }
            Matrix::Banded {
                n,
                ml,
                mu,
                data,
                zero,
            } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                let k = i as isize - j as isize;
                if k < -(*ml as isize) || k > *mu as isize {
                    zero
                } else {
                    let row = (k + *mu as isize) as usize;
                    &data[row * n + j]
                }
            }
        }
    }
}

/// 2D indexing by (i, j): mutable.
///
/// Note:
/// - Identity is immutable and any attempt to mutate via indexing will panic.
/// - For Banded, attempting to mutate outside the band will panic.
impl<T: Real> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        loop {
            // Fast path: full matrix
            if let Matrix::Full { n, data } = self {
                let n_val = *n;
                assert!(i < n_val && j < n_val, "Index out of bounds");
                return &mut data[i * n_val + j];
            }

            // Identity: convert to full, then loop
            if let Matrix::Identity { n, .. } = self {
                let n_val = *n;
                let mut data = vec![T::zero(); n_val * n_val];
                for d in 0..n_val {
                    data[d * n_val + d] = T::one();
                }
                *self = Matrix::Full { n: n_val, data };
                continue;
            }

            // Banded
            if let Matrix::Banded { n, ml, mu, .. } = self {
                let n_val = *n;
                let ml_val = *ml;
                let mu_val = *mu;
                assert!(i < n_val && j < n_val, "Index out of bounds");
                let k = i as isize - j as isize;
                if k >= -(ml_val as isize) && k <= mu_val as isize {
                    // In-band write
                    if let Matrix::Banded { n, mu, data, .. } = self {
                        let n2 = *n;
                        let row = (k + *mu as isize) as usize;
                        return &mut data[row * n2 + j];
                    }
                } else {
                    // Convert to full
                    let old = core::mem::replace(
                        self,
                        Matrix::Identity {
                            n: 0,
                            zero: T::zero(),
                            one: T::one(),
                        },
                    );
                    if let Matrix::Banded {
                        n: n_old,
                        ml: ml_old,
                        mu: mu_old,
                        data,
                        ..
                    } = old
                    {
                        let mut dense = vec![T::zero(); n_old * n_old];
                        for ii in 0..n_old {
                            let j_min = ii.saturating_sub(ml_old);
                            let j_max = (ii + mu_old).min(n_old - 1);
                            for jj in j_min..=j_max {
                                let kk = ii as isize - jj as isize;
                                let row = (kk + mu_old as isize) as usize;
                                dense[ii * n_old + jj] = data[row * n_old + jj];
                            }
                        }
                        *self = Matrix::Full {
                            n: n_old,
                            data: dense,
                        };
                        continue;
                    }
                }
            }

            unreachable!();
        }
    }
}

impl<T> Display for Matrix<T>
where
    T: Real + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.n();
        for i in 0..n {
            f.write_str("[")?;
            for j in 0..n {
                if j > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "{}", self[(i, j)])?;
            }
            f.write_str("]")?;
            if i + 1 < n {
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn identity_converts_to_full_on_write() {
        let mut m: Matrix<f64> = Matrix::identity(3);
        // Initially Identity
        match m {
            Matrix::Identity { .. } => {}
            _ => panic!("expected Identity"),
        }
        // Write off-diagonal -> convert
        m[(2, 0)] = 5.0;
        match m {
            Matrix::Full { .. } => {}
            _ => panic!("expected Full after write"),
        }
        assert_eq!(m[(2, 0)], 5.0);
        assert_eq!(m[(1, 1)], 1.0);
    }

    #[test]
    fn banded_inband_write_keeps_banded() {
        let mut b: Matrix<f64> = Matrix::banded(5, 1, 2);
        b[(2, 2)] = 10.0; // main diag
        b[(1, 2)] = 5.0; // upper-1
        b[(3, 2)] = 7.0; // lower-1
        match b {
            Matrix::Banded { .. } => {}
            _ => panic!("expected Banded to remain banded on in-band writes"),
        }
        assert_eq!(b[(2, 2)], 10.0);
        assert_eq!(b[(1, 2)], 5.0);
        assert_eq!(b[(3, 2)], 7.0);
        // Outside band reads as zero
        assert_eq!(b[(0, 4)], 0.0);
    }

    #[test]
    fn banded_out_of_band_write_converts_to_full() {
        let mut b: Matrix<f64> = Matrix::banded(3, 0, 0); // only main diagonal
        match b {
            Matrix::Banded { .. } => {}
            _ => panic!("expected Banded"),
        }
        b[(0, 2)] = 7.0; // out-of-band -> convert
        match b {
            Matrix::Full { .. } => {}
            _ => panic!("expected Full after out-of-band write"),
        }
        assert_eq!(b[(0, 2)], 7.0);
        // Main diagonal still zero except (0,0) was zero, not identity
        assert_eq!(b[(1, 1)], 0.0);
    }

    #[test]
    fn full_index_read_write() {
        let mut m: Matrix<f64> = Matrix::zeros(2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
    }

    #[test]
    fn display_prints_dense_matrix() {
        let m: Matrix<f64> = Matrix::identity(3);
        let s = format!("{}", m);
        assert!(s.contains("[1 0 0]"));
        assert!(s.contains("[0 1 0]"));
        assert!(s.contains("[0 0 1]"));
    }
}
