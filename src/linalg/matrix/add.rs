//! Matrix addition.

use core::ops::Add;
use core::ops::AddAssign;

use crate::traits::Real;

use super::base::{Matrix, MatrixStorage};

// Add-assign by value
impl<T: Real> AddAssign<Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: Matrix<T>) {
        let n = self.n;
        let m = self.m;
        let lhs = core::mem::replace(self, Matrix::zeros(n, m));
        *self = lhs + rhs;
    }
}

// Matrix + Matrix (elementwise). If both are banded, keep banded with widened bandwidth; otherwise densify.
impl<T: Real> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert_eq!(self.n, rhs.n, "dimension mismatch in Matrix + Matrix");
        let n = self.n;
        match (self, rhs) {
            (
                Matrix {
                    n: n1,
                    m: _,
                    data: _,
                    storage: MatrixStorage::Identity,
                },
                Matrix {
                    n: n2,
                    m: _,
                    data: _,
                    storage: MatrixStorage::Identity,
                },
            ) => {
                assert_eq!(n1, n2);
                let mut data = vec![T::zero(); n * n];
                for i in 0..n {
                    data[i * n + i] = T::one() + T::one();
                }
                Matrix {
                    n: n,
                    m: n,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
            (
                Matrix {
                    data: a,
                    storage: MatrixStorage::Full,
                    ..
                },
                Matrix {
                    data: b,
                    storage: MatrixStorage::Full,
                    ..
                },
            ) => {
                let data = a
                    .into_iter()
                    .zip(b.into_iter())
                    .map(|(x, y)| x + y)
                    .collect();
                Matrix {
                    n: n,
                    m: n,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
            (
                Matrix {
                    data: a,
                    storage: MatrixStorage::Banded { ml, mu, .. },
                    ..
                },
                Matrix {
                    data: b,
                    storage:
                        MatrixStorage::Banded {
                            ml: ml2, mu: mu2, ..
                        },
                    ..
                },
            ) => {
                let ml_out = ml.max(ml2);
                let mu_out = mu.max(mu2);
                let rows_out = ml_out + mu_out + 1;
                let mut out = Matrix {
                    n: n,
                    m: n,
                    data: vec![T::zero(); rows_out * n],
                    storage: MatrixStorage::Banded {
                        ml: ml_out,
                        mu: mu_out,
                        zero: T::zero(),
                    },
                };
                // First input accumulate
                for j in 0..n {
                    for r in 0..(ml + mu + 1) {
                        let k = r as isize - mu as isize;
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < n {
                            let row_out = (k + mu_out as isize) as usize;
                            out.data[row_out * n + j] = out.data[row_out * n + j] + a[r * n + j];
                        }
                    }
                }
                // Second input accumulate
                for j in 0..n {
                    for r in 0..(ml2 + mu2 + 1) {
                        let k = r as isize - mu2 as isize;
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < n {
                            let row_out = (k + mu_out as isize) as usize;
                            out.data[row_out * n + j] = out.data[row_out * n + j] + b[r * n + j];
                        }
                    }
                }
                out
            }
            // Mixed: densify
            (
                Matrix {
                    data: a,
                    storage: sa,
                    ..
                },
                Matrix {
                    data: b,
                    storage: sb,
                    ..
                },
            ) => {
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
                                    let k = r as isize - mu as isize;
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
                let aa = to_full(n, a, sa);
                let bb = to_full(n, b, sb);
                let data = aa
                    .into_iter()
                    .zip(bb.into_iter())
                    .map(|(x, y)| x + y)
                    .collect();
                Matrix {
                    n: n,
                    m: n,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
        }
    }
}

impl<T: Real> Matrix<T> {
    /// Return a new matrix where each stored entry has `rhs` added. Off-band for banded becomes dense if rhs != 0.
    pub fn component_add(mut self, rhs: T) -> Self {
        match &mut self.storage {
            MatrixStorage::Identity => {
                // I + c -> Full with diag 1+c and off-diag c
                let n = self.n;
                let mut data = vec![rhs; n * n];
                for i in 0..n {
                    data[i * n + i] = rhs + T::one();
                }
                Matrix {
                    n: n,
                    m: n,
                    data,
                    storage: MatrixStorage::Full,
                }
            }
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v = *v + rhs;
                }
                self
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let n = self.n;
                if rhs == T::zero() {
                    self
                } else {
                    let rows = *ml + *mu + 1;
                    let mut dense = vec![rhs; n * n];
                    for j in 0..n {
                        for r in 0..rows {
                            let k = r as isize - *mu as isize;
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let i = i_signed as usize;
                                let val = self.data[r * n + j];
                                dense[i * n + j] = val + rhs;
                            }
                        }
                    }
                    Matrix {
                        n: n,
                        m: n,
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
    fn add_scalar_full() {
        let mut m: Matrix<f64> = Matrix::full(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;
        let r = m.component_add(1.0);
        assert_eq!(r[(0, 0)], 2.0);
        assert_eq!(r[(0, 1)], 3.0);
        assert_eq!(r[(1, 0)], 4.0);
        assert_eq!(r[(1, 1)], 5.0);
    }

    #[test]
    fn add_scalar_banded_zero_keeps_banded() {
        let m: Matrix<f64> = Matrix::banded(3, 1, 1);
        let r = m.component_add(0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(r[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn add_matrix_full_full() {
        let mut a: Matrix<f64> = Matrix::full(2, 2);
        a[(0, 0)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 3.0;
        a[(1, 1)] = 4.0;
        let mut b: Matrix<f64> = Matrix::full(2, 2);
        b[(0, 0)] = 4.0;
        b[(0, 1)] = 3.0;
        b[(1, 0)] = 2.0;
        b[(1, 1)] = 1.0;
        let r = a + b;
        assert_eq!(r[(0, 0)], 5.0);
        assert_eq!(r[(0, 1)], 5.0);
        assert_eq!(r[(1, 0)], 5.0);
        assert_eq!(r[(1, 1)], 5.0);
    }

    #[test]
    fn add_matrix_banded_banded() {
        // 3x3, ml=1, mu=0 (lower tri without main above)
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
        let r = a + b;
        // Check entries of the resulting tri-diagonal
        assert_eq!(r[(0, 0)], 3.0);
        assert_eq!(r[(1, 1)], 3.0);
        assert_eq!(r[(2, 2)], 3.0);
        assert_eq!(r[(1, 0)], 1.0);
        assert_eq!(r[(2, 1)], 1.0);
        assert_eq!(r[(0, 1)], 2.0);
        assert_eq!(r[(1, 2)], 2.0);
        assert_eq!(r[(0, 2)], 0.0);
    }
}
