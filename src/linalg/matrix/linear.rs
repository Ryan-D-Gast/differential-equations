//! Linear system solve: solve A x = b using LU decomposition on a dense copy.

use crate::traits::{Real, State};

use super::base::SquareMatrix;

impl<T: Real> SquareMatrix<T> {
    /// Solve A x = b for x using LU decomposition with partial pivoting.
    /// Returns the solution vector with the same State<T> type as b.
    /// Internally densifies A if needed.
    pub fn lin_solve<V>(&self, b: V) -> V
    where
        V: State<T>,
    {
        let n = self.n();
        assert_eq!(
            b.len(),
            n,
            "dimension mismatch in solve: A is {}x{}, b has length {}",
            n,
            n,
            b.len()
        );

        // 1) Densify A into a Vec<T> of size n*n (row-major)
        let mut a = vec![T::zero(); n * n];
        match self {
            SquareMatrix::Identity { .. } => {
                for i in 0..n {
                    a[i * n + i] = T::one();
                }
            }
            SquareMatrix::Full { n: nn, data } => {
                let nnv = *nn;
                a.copy_from_slice(&data[0..nnv * nnv]);
            }
            SquareMatrix::Banded {
                n: nn,
                ml,
                mu,
                data,
                ..
            } => {
                let rows = *ml + *mu + 1;
                let nnv = *nn;
                for j in 0..nnv {
                    for r in 0..rows {
                        let k = r as isize - *mu as isize; // i - j
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < nnv {
                            let i = i_signed as usize;
                            a[i * nnv + j] = a[i * nnv + j] + data[r * nnv + j];
                        }
                    }
                }
            }
        }

        // 2) Copy b into a dense vector x and perform in-place solve
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[i] = b.get(i);
        }

        // 3) LU factorization with partial pivoting (Doolittle-style)
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            // Find pivot row
            let mut pivot_row = k;
            let mut pivot_val = a[k * n + k].abs();
            for i in (k + 1)..n {
                let val = a[i * n + k].abs();
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val == T::zero() {
                panic!("singular matrix in solve");
            }
            if pivot_row != k {
                // swap rows in A
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
                // swap entries in x (we'll apply permutation to RHS)
                x.swap(k, pivot_row);
                piv.swap(k, pivot_row);
            }
            // Eliminate below
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / akk;
                a[i * n + k] = factor; // store L(i,k)
                for j in (k + 1)..n {
                    a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
                }
            }
        }

        // 4) Forward solve Ly = Pb (x currently holds permuted b)
        for i in 0..n {
            let mut sum = x[i];
            for k in 0..i {
                sum = sum - a[i * n + k] * x[k];
            }
            x[i] = sum; // since L has ones on diagonal
        }

        // 5) Backward solve Ux = y
        for i in (0..n).rev() {
            let mut sum = x[i];
            for k in (i + 1)..n {
                sum = sum - a[i * n + k] * x[k];
            }
            x[i] = sum / a[i * n + i];
        }

        // 6) Build output State from x
        let mut out = V::zeros();
        for i in 0..n {
            out.set(i, x[i]);
        }
        out
    }

    /// Solve A x = b in-place on the provided slice b (overwritten with x).
    /// Uses LU with partial pivoting on a dense copy of A.
    pub fn lin_solve_in_place(&self, b: &mut [T]) {
        let n = self.n();
        assert_eq!(
            b.len(),
            n,
            "dimension mismatch in solve: A is {}x{}, b has length {}",
            n,
            n,
            b.len()
        );

        // Densify A into row-major Vec<T>
        let mut a = vec![T::zero(); n * n];
        match self {
            SquareMatrix::Identity { .. } => {
                for i in 0..n {
                    a[i * n + i] = T::one();
                }
            }
            SquareMatrix::Full { n: nn, data } => {
                let nnv = *nn;
                a.copy_from_slice(&data[0..nnv * nnv]);
            }
            SquareMatrix::Banded {
                n: nn,
                ml,
                mu,
                data,
                ..
            } => {
                let rows = *ml + *mu + 1;
                let nnv = *nn;
                for j in 0..nnv {
                    for r in 0..rows {
                        let k = r as isize - *mu as isize;
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < nnv {
                            let i = i_signed as usize;
                            a[i * nnv + j] = a[i * nnv + j] + data[r * nnv + j];
                        }
                    }
                }
            }
        }

        // LU factorization with partial pivoting, applying permutations to b
        for k in 0..n {
            // pivot
            let mut pivot_row = k;
            let mut pivot_val = a[k * n + k].abs();
            for i in (k + 1)..n {
                let val = a[i * n + k].abs();
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val == T::zero() {
                panic!("singular matrix in solve");
            }
            if pivot_row != k {
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
                b.swap(k, pivot_row);
            }
            // eliminate
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / akk;
                a[i * n + k] = factor;
                for j in (k + 1)..n {
                    a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
                }
            }
        }

        // Forward solve Ly = Pb (b currently permuted) stored back into b
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum = sum - a[i * n + k] * b[k];
            }
            b[i] = sum;
        }
        // Backward solve Ux = y
        for i in (0..n).rev() {
            let mut sum = b[i];
            for k in (i + 1)..n {
                sum = sum - a[i * n + k] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SquareMatrix;
    use nalgebra::Vector2;

    #[test]
    fn solve_full_2x2() {
        // A = [[3, 2],[1, 4]], b = [5, 6] -> x = [0.8, 1.3]
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![3.0, 2.0, 1.0, 4.0]);
        let b = Vector2::new(5.0, 6.0);
        let x = a.lin_solve(b);
        // Solve manually: [[3,2],[1,4]] x = [5,6] => x = [ (20-12)/10, (15-5)/10 ] = [0.8, 1.3]
        assert!((x.x - 0.8).abs() < 1e-12);
        assert!((x.y - 1.3).abs() < 1e-12);
    }
}
