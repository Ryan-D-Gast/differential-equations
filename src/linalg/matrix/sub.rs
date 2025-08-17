//! Matrix subtraction.

use core::ops::{Sub, SubAssign};

use super::base::{Matrix, MatrixStorage};
use crate::traits::Real;

// Matrix - Matrix
impl<T: Real> Sub for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        match (self, rhs) {
            (
                Matrix {
                    nrows: n1,
                    storage: MatrixStorage::Identity,
                    ..
                },
                Matrix {
                    nrows: n2,
                    storage: MatrixStorage::Identity,
                    ..
                },
            ) => {
                assert_eq!(n1, n2, "dimension mismatch in Matrix - Matrix");
                Matrix {
                    nrows: n1,
                    ncols: n1,
                    data: vec![T::zero(); n1 * n1],
                    storage: MatrixStorage::Full,
                }
            }
            (
                Matrix {
                    nrows: n,
                    data: mut a,
                    storage: MatrixStorage::Full,
                    ..
                },
                Matrix {
                    nrows: n2,
                    data: b,
                    storage: MatrixStorage::Full,
                    ..
                },
            ) => {
                assert_eq!(n, n2, "dimension mismatch in Matrix - Matrix");
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = *x - *y;
                }
                Matrix {
                    nrows: n,
                    ncols: n,
                    data: a,
                    storage: MatrixStorage::Full,
                }
            }
            (
                Matrix {
                    nrows: n,
                    data: a,
                    storage: MatrixStorage::Banded { ml, mu, .. },
                    ..
                },
                Matrix {
                    nrows: n2,
                    data: b,
                    storage:
                        MatrixStorage::Banded {
                            ml: ml2, mu: mu2, ..
                        },
                    ..
                },
            ) => {
                assert_eq!(n, n2, "dimension mismatch in Matrix - Matrix");
                let ml_out = ml.max(ml2);
                let mu_out = mu.max(mu2);
                let rows_out = ml_out + mu_out + 1;
                let mut out = Matrix {
                    nrows: n,
                    ncols: n,
                    data: vec![T::zero(); rows_out * n],
                    storage: MatrixStorage::Banded {
                        ml: ml_out,
                        mu: mu_out,
                        zero: T::zero(),
                    },
                };
                // Add first banded
                for j in 0..n {
                    for r in 0..(ml + mu + 1) {
                        let k = r as isize - mu as isize; // i - j for first
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < n {
                            let row_out = (k + mu_out as isize) as usize;
                            out.data[row_out * n + j] = out.data[row_out * n + j] + a[r * n + j];
                        }
                    }
                }
                // Subtract second banded
                for j in 0..n {
                    for r in 0..(ml2 + mu2 + 1) {
                        let k = r as isize - mu2 as isize; // i - j for second
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < n {
                            let row_out = (k + mu_out as isize) as usize;
                            out.data[row_out * n + j] = out.data[row_out * n + j] - b[r * n + j];
                        }
                    }
                }
                out
            }
            // Mixed storage: densify
            (
                Matrix {
                    nrows: n1,
                    data: a,
                    storage: sa,
                    ..
                },
                Matrix {
                    nrows: n2,
                    data: b,
                    storage: sb,
                    ..
                },
            ) => {
                assert_eq!(n1, n2, "dimension mismatch in Matrix - Matrix");
                let to_full = |n: usize, data: Vec<T>, storage: MatrixStorage<T>| -> Vec<T> {
                    match storage {
                        MatrixStorage::Full => data,
                        MatrixStorage::Identity => {
                            let mut d = vec![T::zero(); n * n];
                            for i in 0..n {
                                d[i * n + i] = T::one();
                            }
                            d
                        }
                        MatrixStorage::Banded { ml, mu, .. } => {
                            let mut d = vec![T::zero(); n * n];
                            for j in 0..n {
                                for r in 0..(ml + mu + 1) {
                                    let k = r as isize - mu as isize; // i - j
                                    let i_signed = j as isize + k;
                                    if i_signed >= 0 && (i_signed as usize) < n {
                                        let i = i_signed as usize;
                                        d[i * n + j] = d[i * n + j] + data[r * n + j];
                                    }
                                }
                            }
                            d
                        }
                    }
                };
                let aa = to_full(n1, a, sa);
                let bb = to_full(n2, b, sb);
                let data = aa
                    .into_iter()
                    .zip(bb.into_iter())
                    .map(|(x, y)| x - y)
                    .collect();
                Matrix {
                    nrows: n1,
                    ncols: n1,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
        }
    }
}

// For scalars, use `component_sub`.

// Sub-assign by value
impl<T: Real> SubAssign<Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        let n = self.n();
        let lhs = core::mem::replace(self, Matrix::zeros(n));
        *self = lhs - rhs;
    }
}

// Sub-assign by reference (clones rhs)
impl<T: Real> SubAssign<&Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        let n = self.n();
        let lhs = core::mem::replace(self, Matrix::zeros(n));
        *self = lhs - rhs.clone();
    }
}

impl<T: Real> Matrix<T> {
    /// Return a new matrix where each stored entry has `rhs` subtracted. Off-band handling similar to add.
    pub fn component_sub(mut self, rhs: T) -> Self {
        match &mut self.storage {
            MatrixStorage::Identity => {
                let n = self.nrows;
                let mut data = vec![T::zero() - rhs; n * n];
                for i in 0..n {
                    data[i * n + i] = T::one() - rhs;
                }
                Matrix {
                    nrows: n,
                    ncols: n,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v = *v - rhs;
                }
                self
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let n = self.nrows;
                if rhs == T::zero() {
                    self
                } else {
                    let rows = *ml + *mu + 1;
                    let mut dense = vec![T::zero() - rhs; n * n];
                    for j in 0..n {
                        for r in 0..rows {
                            let k = r as isize - *mu as isize;
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let i = i_signed as usize;
                                let val = self.data[r * n + j];
                                dense[i * n + j] = val - rhs;
                            }
                        }
                    }
                    Matrix {
                        nrows: n,
                        ncols: n,
                        data: dense,
                        storage: MatrixStorage::Full,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrix::Matrix;

    #[test]
    fn sub_scalar_full() {
        let m: Matrix<f64> = Matrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let r = m.component_sub(1.0);
        assert_eq!(r[(0, 0)], 0.0);
        assert_eq!(r[(0, 1)], 1.0);
        assert_eq!(r[(1, 0)], 2.0);
        assert_eq!(r[(1, 1)], 3.0);
    }

    #[test]
    fn sub_scalar_banded_zero_keeps_banded() {
        let m: Matrix<f64> = Matrix::banded(3, 1, 1);
        let r = m.component_sub(0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(r[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn sub_matrix_full_full() {
        let a: Matrix<f64> = Matrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<f64> = Matrix::full(2, vec![4.0, 3.0, 2.0, 1.0]);
        let r = a - b;
        assert_eq!(r[(0, 0)], -3.0);
        assert_eq!(r[(0, 1)], -1.0);
        assert_eq!(r[(1, 0)], 1.0);
        assert_eq!(r[(1, 1)], 3.0);
    }

    #[test]
    fn sub_matrix_banded_banded() {
        // 3x3, ml=1, mu=0 and 0,1
        let mut a: Matrix<f64> = Matrix::banded(3, 1, 0);
        let mut b: Matrix<f64> = Matrix::banded(3, 0, 1);
        // set a main and lower
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 1.0;
        a[(1, 0)] = 1.0;
        a[(2, 1)] = 1.0;
        // set b main and upper
        b[(0, 0)] = 2.0;
        b[(1, 1)] = 2.0;
        b[(2, 2)] = 2.0;
        b[(0, 1)] = 2.0;
        b[(1, 2)] = 2.0;
        let r = a - b;
        // Check entries of the resulting tri-diagonal
        assert_eq!(r[(0, 0)], -1.0);
        assert_eq!(r[(1, 1)], -1.0);
        assert_eq!(r[(2, 2)], -1.0);
        assert_eq!(r[(1, 0)], 1.0);
        assert_eq!(r[(2, 1)], 1.0);
        assert_eq!(r[(0, 1)], -2.0);
        assert_eq!(r[(1, 2)], -2.0);
        assert_eq!(r[(0, 2)], 0.0);
    }
}
