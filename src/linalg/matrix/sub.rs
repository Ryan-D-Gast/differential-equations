//! Subtraction operations for SquareMatrix.

use core::ops::Sub;

use crate::traits::Real;

use super::base::SquareMatrix;

// SquareMatrix - scalar (elementwise subtract constant)
impl<T: Real> Sub<T> for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
            SquareMatrix::Identity { n, .. } => {
                // I - c -> diagonal 1-c, off-diagonals -c
                let mut out = SquareMatrix::Full {
                    n,
                    data: vec![T::zero(); n * n],
                };
                if let SquareMatrix::Full { n, data } = &mut out {
                    for v in data.iter_mut() {
                        *v = T::zero() - rhs;
                    }
                    for i in 0..*n {
                        data[i * *n + i] = T::one() - rhs;
                    }
                }
                out
            }
            SquareMatrix::Full { n, mut data } => {
                for v in &mut data {
                    *v = *v - rhs;
                }
                SquareMatrix::Full { n, data }
            }
            SquareMatrix::Banded {
                n,
                ml,
                mu,
                data,
                zero,
            } => {
                if rhs == T::zero() {
                    SquareMatrix::Banded {
                        n,
                        ml,
                        mu,
                        data,
                        zero,
                    }
                } else {
                    let mut dense = vec![T::zero() - rhs; n * n];
                    let rows = ml + mu + 1;
                    for j in 0..n {
                        for r in 0..rows {
                            let k = r as isize - mu as isize; // i - j
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let i = i_signed as usize;
                                let val = data[r * n + j];
                                dense[i * n + j] = val - rhs;
                            }
                        }
                    }
                    SquareMatrix::Full { n, data: dense }
                }
            }
        }
    }
}

// SquareMatrix - SquareMatrix
impl<T: Real> Sub for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn sub(self, rhs: SquareMatrix<T>) -> Self::Output {
        match (self, rhs) {
            (SquareMatrix::Identity { n: n1, .. }, SquareMatrix::Identity { n: n2, .. }) => {
                assert_eq!(n1, n2, "dimension mismatch in SquareMatrix - SquareMatrix");
                // I - I = zero matrix (dense)
                SquareMatrix::Full {
                    n: n1,
                    data: vec![T::zero(); n1 * n1],
                }
            }
            (SquareMatrix::Full { n, mut data }, SquareMatrix::Full { n: n2, data: data2 }) => {
                assert_eq!(n, n2, "dimension mismatch in SquareMatrix - SquareMatrix");
                for (a, b) in data.iter_mut().zip(data2.iter()) {
                    *a = *a - *b;
                }
                SquareMatrix::Full { n, data }
            }
            (
                SquareMatrix::Banded {
                    n, ml, mu, data, ..
                },
                SquareMatrix::Banded {
                    n: n2,
                    ml: ml2,
                    mu: mu2,
                    data: data2,
                    ..
                },
            ) => {
                assert_eq!(n, n2, "dimension mismatch in SquareMatrix - SquareMatrix");
                let ml_out = ml.max(ml2);
                let mu_out = mu.max(mu2);
                let rows_out = ml_out + mu_out + 1;
                let mut out = SquareMatrix::Banded {
                    n,
                    ml: ml_out,
                    mu: mu_out,
                    data: vec![T::zero(); rows_out * n],
                    zero: T::zero(),
                };
                if let SquareMatrix::Banded {
                    mu: mu_out2,
                    data: out_data,
                    ..
                } = &mut out
                {
                    // Add first banded
                    for j in 0..n {
                        for r in 0..(ml + mu + 1) {
                            let k = r as isize - mu as isize; // i - j for first
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let row_out = (k + *mu_out2 as isize) as usize;
                                out_data[row_out * n + j] =
                                    out_data[row_out * n + j] + data[r * n + j];
                            }
                        }
                    }
                    // Subtract second banded
                    for j in 0..n {
                        for r in 0..(ml2 + mu2 + 1) {
                            let k = r as isize - mu2 as isize; // i - j for second
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let row_out = (k + *mu_out2 as isize) as usize;
                                out_data[row_out * n + j] =
                                    out_data[row_out * n + j] - data2[r * n + j];
                            }
                        }
                    }
                }
                out
            }
            // Mixed cases: densify
            (a, b) => {
                fn to_full<T: Real>(m: SquareMatrix<T>) -> (usize, Vec<T>) {
                    match m {
                        SquareMatrix::Full { n, data } => (n, data),
                        SquareMatrix::Identity { n, .. } => {
                            let mut d = vec![T::zero(); n * n];
                            for i in 0..n {
                                d[i * n + i] = T::one();
                            }
                            (n, d)
                        }
                        SquareMatrix::Banded {
                            n, ml, mu, data, ..
                        } => {
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
                            (n, d)
                        }
                    }
                }
                let (n1, mut a) = to_full(a);
                let (n2, b) = to_full(b);
                assert_eq!(n1, n2, "dimension mismatch in SquareMatrix - SquareMatrix");
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = *x - *y;
                }
                SquareMatrix::Full { n: n1, data: a }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SquareMatrix;

    #[test]
    fn sub_scalar_full() {
        let m: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let r = m - 1.0;
        match r {
            SquareMatrix::Full { data, .. } => assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0]),
            _ => panic!("expected full"),
        }
    }

    #[test]
    fn sub_scalar_banded_zero_keeps_banded() {
        let m: SquareMatrix<f64> = SquareMatrix::banded(3, 1, 1);
        let r = m - 0.0;
        match r {
            SquareMatrix::Banded { .. } => {}
            _ => panic!("expected banded"),
        }
    }

    #[test]
    fn sub_matrix_full_full() {
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: SquareMatrix<f64> = SquareMatrix::full(2, vec![4.0, 3.0, 2.0, 1.0]);
        let r = a - b;
        match r {
            SquareMatrix::Full { data, .. } => assert_eq!(data, vec![-3.0, -1.0, 1.0, 3.0]),
            _ => panic!("expected full"),
        }
    }

    #[test]
    fn sub_matrix_banded_banded() {
        // 3x3, ml=1, mu=0 and 0,1
        let mut a: SquareMatrix<f64> = SquareMatrix::banded(3, 1, 0);
        let mut b: SquareMatrix<f64> = SquareMatrix::banded(3, 0, 1);
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
        // Bands combined -> ml=1, mu=1
        match r {
            SquareMatrix::Banded { ml, mu, .. } => {
                assert_eq!(ml, 1);
                assert_eq!(mu, 1);
            }
            _ => panic!("expected banded"),
        }
    }
}
