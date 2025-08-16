//! Addition operations for SquareMatrix.

use core::ops::Add;
use core::ops::{AddAssign};

use crate::traits::Real;

use super::base::SquareMatrix;

// SquareMatrix + scalar (elementwise add of a constant)
impl<T: Real> Add<T> for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn add(self, rhs: T) -> Self::Output {
        match self {
            SquareMatrix::Identity { n, zero: _, one: _ } => {
                // I + c -> diagonal with 1+c, off-diagonals c
                let mut out = SquareMatrix::Full {
                    n,
                    data: vec![T::zero(); n * n],
                };
                if let SquareMatrix::Full { n, data } = &mut out {
                    // fill with rhs
                    for v in data.iter_mut() {
                        *v = rhs;
                    }
                    // add 1 to diagonal
                    for i in 0..*n {
                        data[i * *n + i] = data[i * *n + i] + T::one();
                    }
                }
                out
            }
            SquareMatrix::Full { n, mut data } => {
                for v in &mut data {
                    *v = *v + rhs;
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
                // Adding a scalar c to a banded matrix produces a full matrix unless c == 0,
                // because off-band zeros become c.
                if rhs == T::zero() {
                    SquareMatrix::Banded {
                        n,
                        ml,
                        mu,
                        data,
                        zero,
                    }
                } else {
                    let mut dense = vec![rhs; n * n];
                    let rows = ml + mu + 1;
                    for j in 0..n {
                        // iterate storage rows
                        for r in 0..rows {
                            let k = r as isize - mu as isize; // i - j
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let i = i_signed as usize;
                                let val = data[r * n + j];
                                dense[i * n + j] = val + rhs; // overwrite in-band with m_ij + c
                            }
                        }
                    }
                    SquareMatrix::Full { n, data: dense }
                }
            }
        }
    }
}

// Add-assign: self += scalar
impl<T: Real> AddAssign<T> for SquareMatrix<T> {
    fn add_assign(&mut self, rhs: T) {
        let n = self.n();
        let lhs = core::mem::replace(self, SquareMatrix::zeros(n));
        *self = lhs + rhs;
    }
}

// Add-assign: self += matrix (by value)
impl<T: Real> AddAssign<SquareMatrix<T>> for SquareMatrix<T> {
    fn add_assign(&mut self, rhs: SquareMatrix<T>) {
        let n = self.n();
        let lhs = core::mem::replace(self, SquareMatrix::zeros(n));
        *self = lhs + rhs;
    }
}

// Add-assign: self += &matrix (by reference, clones rhs)
impl<T: Real> AddAssign<&SquareMatrix<T>> for SquareMatrix<T> {
    fn add_assign(&mut self, rhs: &SquareMatrix<T>) {
        let n = self.n();
        let lhs = core::mem::replace(self, SquareMatrix::zeros(n));
        *self = lhs + rhs.clone();
    }
}
// SquareMatrix + SquareMatrix (elementwise), result shape rules:
// If both banded, and the sum of bands covers only a band (i.e., ml=max, mu=max), keep banded; else become Full.
impl<T: Real> Add for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn add(self, rhs: SquareMatrix<T>) -> Self::Output {
        match (self, rhs) {
            (SquareMatrix::Identity { n: n1, .. }, SquareMatrix::Identity { n: n2, .. }) => {
                assert_eq!(n1, n2, "dimension mismatch in SquareMatrix + SquareMatrix");
                // I + I = diag(2)
                let mut data = vec![T::zero(); n1 * n1];
                for i in 0..n1 {
                    data[i * n1 + i] = T::one() + T::one();
                }
                SquareMatrix::Full { n: n1, data }
            }
            (SquareMatrix::Full { n, mut data }, SquareMatrix::Full { n: n2, data: data2 }) => {
                assert_eq!(n, n2, "dimension mismatch in SquareMatrix + SquareMatrix");
                for (a, b) in data.iter_mut().zip(data2.iter()) {
                    *a = *a + *b;
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
                assert_eq!(n, n2, "dimension mismatch in SquareMatrix + SquareMatrix");
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
                    // Add first input
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
                    // Add second input
                    for j in 0..n {
                        for r in 0..(ml2 + mu2 + 1) {
                            let k = r as isize - mu2 as isize; // i - j for second
                            let i_signed = j as isize + k;
                            if i_signed >= 0 && (i_signed as usize) < n {
                                let row_out = (k + *mu_out2 as isize) as usize;
                                out_data[row_out * n + j] =
                                    out_data[row_out * n + j] + data2[r * n + j];
                            }
                        }
                    }
                }
                out
            }
            // Mixed cases: convert to Full and add
            (a, b) => {
                // Helper to densify
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
                assert_eq!(n1, n2, "dimension mismatch in SquareMatrix + SquareMatrix");
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = *x + *y;
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
    fn add_scalar_full() {
        let m: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let r = m + 1.0;
        match r {
            SquareMatrix::Full { data, .. } => assert_eq!(data, vec![2.0, 3.0, 4.0, 5.0]),
            _ => panic!("expected full"),
        }
    }

    #[test]
    fn add_scalar_banded_zero_keeps_banded() {
        let m: SquareMatrix<f64> = SquareMatrix::banded(3, 1, 1);
        let r = m + 0.0;
        match r {
            SquareMatrix::Banded { .. } => {}
            _ => panic!("expected banded"),
        }
    }

    #[test]
    fn add_matrix_full_full() {
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: SquareMatrix<f64> = SquareMatrix::full(2, vec![4.0, 3.0, 2.0, 1.0]);
        let r = a + b;
        match r {
            SquareMatrix::Full { data, .. } => assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]),
            _ => panic!("expected full"),
        }
    }

    #[test]
    fn add_matrix_banded_banded() {
        // 3x3, ml=1, mu=0 (lower tri without main above)
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
        let r = a + b;
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
