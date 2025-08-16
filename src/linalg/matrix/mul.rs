//! Multiplication for SquareMatrix: scalar, vector (State), and matrix product.

use core::ops::Mul;

use crate::traits::{Real, State};

use super::base::SquareMatrix;

// SquareMatrix * Vector (State)
impl<T, V> Mul<V> for SquareMatrix<T>
where
    T: Real,
    V: State<T>,
{
    type Output = V;

    fn mul(self, rhs: V) -> Self::Output {
        let n = self.n();
        assert_eq!(rhs.len(), n, "dimension mismatch in SquareMatrix * Vector");
        // Accumulate into a fresh vector-like State using zeros and set
        let mut out = V::zeros();
        // Helper to add into out[i]
        match self {
            SquareMatrix::Identity { .. } => {
                // out = rhs
                for i in 0..n {
                    out.set(i, rhs.get(i));
                }
            }
            SquareMatrix::Full { n, data } => {
                for i in 0..n {
                    let mut acc = T::zero();
                    for j in 0..n {
                        acc = acc + data[i * n + j] * rhs.get(j);
                    }
                    out.set(i, acc);
                }
            }
            SquareMatrix::Banded {
                n, ml, mu, data, ..
            } => {
                let rows = ml + mu + 1;
                for j in 0..n {
                    for r in 0..rows {
                        let k = r as isize - mu as isize; // i - j
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < n {
                            let i = i_signed as usize;
                            let prev = out.get(i);
                            let add = data[r * n + j] * rhs.get(j);
                            out.set(i, prev + add);
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::SquareMatrix;
    use nalgebra::Vector2;
    use num_complex::Complex;

    #[test]
    fn mul_matrix_vector_full() {
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let v = Vector2::new(5.0, 6.0);
        let out = a * v;
        assert_eq!(out.x, 17.0);
        assert_eq!(out.y, 39.0);
    }

    #[test]
    fn mul_identity_vector() {
        let a: SquareMatrix<f64> = SquareMatrix::identity(2);
        let v: Complex<f64> = Complex::new(5.0, 6.0);
        let out = a * v;
        assert_eq!(out.re, 5.0);
        assert_eq!(out.im, 6.0);
    }
}
