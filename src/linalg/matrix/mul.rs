//! Multiplication for SquareMatrix: scalar, vector (State), and matrix product.

use core::ops::{Mul, MulAssign};

use crate::traits::Real;

use super::base::SquareMatrix;

// SquareMatrix * scalar (elementwise scale)
impl<T> Mul<T> for SquareMatrix<T>
where
    T: Real,
{
    type Output = SquareMatrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        match self {
            // a * I -> diagonal matrix with `a` on the diagonal
            SquareMatrix::Identity { n, .. } => SquareMatrix::diagonal(vec![rhs; n]),
            // Scale each entry in the dense storage
            SquareMatrix::Full { n, data } => {
                SquareMatrix::full(n, data.into_iter().map(|x| x * rhs).collect())
            }
            // Scale each stored band entry; preserve bandwidths
            SquareMatrix::Banded { n, ml, mu, data, .. } => SquareMatrix::Banded {
                n,
                ml,
                mu,
                data: data.into_iter().map(|x| x * rhs).collect(),
                zero: T::zero(),
            },
        }
    }
}

// In-place scalar scale: self *= scalar
impl<T> MulAssign<T> for SquareMatrix<T>
where
    T: Real,
{
    fn mul_assign(&mut self, rhs: T) {
        let n = self.n();
        let lhs = core::mem::replace(self, SquareMatrix::zeros(n));
        *self = lhs * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::SquareMatrix;

    #[test]
    fn mul_matrix_full() {
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = 5.0;
        let out = a * s;
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 10.0);
        assert_eq!(out[(1, 0)], 15.0);
        assert_eq!(out[(1, 1)], 20.0);
    }

    #[test]
    fn mul_identity() {
        let a: SquareMatrix<f64> = SquareMatrix::identity(2);
        let s = 5.0;
        let out = a * s;
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 0.0);
        assert_eq!(out[(1, 0)], 0.0);
        assert_eq!(out[(1, 1)], 5.0);
    }

    #[test]
    fn mul_assign() {
        let mut a: SquareMatrix<f64> = SquareMatrix::identity(2);
        let s = 5.0;
        a *= s;
        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], 0.0);
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(1, 1)], 5.0);
    }
}
