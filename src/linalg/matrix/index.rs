//! Indexing and Display implementations.

use core::fmt::{self, Display, Write as _};
use core::ops::{Index, IndexMut};

use crate::traits::Real;

use super::base::SquareMatrix;

/// 2D indexing by (i, j): read-only.
impl<T: Real> Index<(usize, usize)> for SquareMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        match self {
            SquareMatrix::Identity { n, zero, one } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                if i == j { one } else { zero }
            }
            SquareMatrix::Full { n, data } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                &data[i * n + j]
            }
            SquareMatrix::Banded {
                n,
                ml,
                mu,
                data,
                zero,
            } => {
                assert!(i < *n && j < *n, "Index out of bounds");
                let k = i as isize - j as isize;
                // In-band condition: -mu <= i - j <= ml
                if k < -(*mu as isize) || k > *ml as isize {
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
impl<T: Real> IndexMut<(usize, usize)> for SquareMatrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        match self {
            SquareMatrix::Full { n, data } => {
                let n_val = *n;
                assert!(i < n_val && j < n_val, "Index out of bounds");
                &mut data[i * n_val + j]
            }
            SquareMatrix::Identity { .. } => {
                panic!(
                    "cannot mutate Identity matrix via indexing; convert explicitly to Full first"
                )
            }
            SquareMatrix::Banded {
                n, ml, mu, data, ..
            } => {
                let n_val = *n;
                assert!(i < n_val && j < n_val, "Index out of bounds");
                let k = i as isize - j as isize;
                // In-band condition: -mu <= i - j <= ml
                if k >= -(*mu as isize) && k <= *ml as isize {
                    let row = (k + *mu as isize) as usize;
                    &mut data[row * n_val + j]
                } else {
                    panic!(
                        "attempted to write outside band of Banded matrix: i-j={} not in [-mu, ml] = [-{}, {}]",
                        k, mu, ml
                    )
                }
            }
        }
    }
}

impl<T> Display for SquareMatrix<T>
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
    use super::SquareMatrix;

    #[test]
    #[should_panic(expected = "cannot mutate Identity matrix via indexing")]
    fn identity_panics_on_write() {
        let mut m: SquareMatrix<f64> = SquareMatrix::identity(3);
        m[(2, 0)] = 5.0; // should panic
    }

    #[test]
    fn banded_inband_write_keeps_banded() {
        let mut b: SquareMatrix<f64> = SquareMatrix::banded(5, 1, 2);
        b[(2, 2)] = 10.0; // main diag
        b[(1, 2)] = 5.0; // upper-1
        b[(3, 2)] = 7.0; // lower-1
        match b {
            SquareMatrix::Banded { .. } => {}
            _ => panic!("expected Banded to remain banded on in-band writes"),
        }
        assert_eq!(b[(2, 2)], 10.0);
        assert_eq!(b[(1, 2)], 5.0);
        assert_eq!(b[(3, 2)], 7.0);
        // Outside band reads as zero
        assert_eq!(b[(0, 4)], 0.0);
    }

    #[test]
    #[should_panic(expected = "attempted to write outside band of Banded matrix")]
    fn banded_out_of_band_write_panics() {
        let mut b: SquareMatrix<f64> = SquareMatrix::banded(3, 0, 0); // only main diagonal
        b[(0, 2)] = 7.0; // out-of-band -> panic
    }

    #[test]
    fn full_index_read_write() {
        let mut m: SquareMatrix<f64> = SquareMatrix::zeros(2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
    }

    #[test]
    fn display_prints_dense_matrix() {
        let m: SquareMatrix<f64> = SquareMatrix::identity(3);
        let s = format!("{}", m);
        assert!(s.contains("[1 0 0]"));
        assert!(s.contains("[0 1 0]"));
        assert!(s.contains("[0 0 1]"));
    }
}
