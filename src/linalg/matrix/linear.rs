//! Linear solves: A x = b via LU with partial pivoting on a dense copy.

use crate::{
    linalg::LinalgError,
    traits::{Real, State},
};

use super::base::Matrix;

impl<T: Real> Matrix<T> {
    /// Solve A x = b, Returns Err if the matrix is singular or dimensions are incompatible
    pub fn lin_solve<Y>(&self, b: Y) -> Result<Y, LinalgError>
    where
        Y: State<T>,
    {
        let n = self.n;
        if self.m != n {
            return Err(LinalgError::BadInput {
                message: "Matrix solve requires a square matrix".into(),
            });
        }
        if b.len() != n {
            return Err(LinalgError::BadInput {
                message: "Incompatible vector length".into(),
            });
        }

        // 1) Densify A into a Vec<T> of size n*n (row-major)
        let mut a = self.to_dense_vec();

        // 2) Use b as the working state
        let mut x = b.clone();

        // 3) LU factorization with partial pivoting and singularity checking
        let mut piv: Vec<usize> = (0..n).collect();
        let eps = T::from_f64(1e-14).unwrap(); // Singularity threshold

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

            // Check for singularity
            if pivot_val <= eps {
                // Note the t, y are not known here and should be updated by caller before returning to user
                return Err(LinalgError::Singular { step: k + 1 });
            }

            if pivot_row != k {
                // swap rows in A
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
                // swap entries in x
                let tk = x.get_component(k);
                let tm = x.get_component(pivot_row);
                x.set_component(k, tm);
                x.set_component(pivot_row, tk);
                piv.swap(k, pivot_row);
            }

            // Eliminate below the pivot
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / akk;
                a[i * n + k] = factor; // store L(i,k)
                for j in (k + 1)..n {
                    a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
                }
            }
        }

        // Forward solve Ly = Pb (x currently holds permuted b)
        for i in 0..n {
            let mut sum = x.get_component(i);
            for k in 0..i {
                sum -= a[i * n + k] * x.get_component(k);
            }
            x.set_component(i, sum); // since L has ones on diagonal
        }

        // Backward solve Ux = y
        for i in (0..n).rev() {
            let mut sum = x.get_component(i);
            for k in (i + 1)..n {
                sum -= a[i * n + k] * x.get_component(k);
            }
            x.set_component(i, sum / a[i * n + i]);
        }

        Ok(x)
    }

    /// In-place solve: overwrites `b` with `x`.
    pub fn lin_solve_mut(&self, b: &mut [T]) -> Result<(), LinalgError> {
        let n = self.n;
        assert_eq!(
            self.m, n,
            "dimension mismatch in solve: A must be square, got {}x{}",
            n, self.m
        );
        assert_eq!(
            b.len(),
            n,
            "dimension mismatch in solve: A is {}x{}, b has length {}",
            n,
            n,
            b.len()
        );

        // Densify A into row-major Vec<T>
        let mut a = self.to_dense_vec();

        // LU with partial pivoting, applying permutations to b
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
                return Err(LinalgError::Singular { step: k + 1 });
            }
            if pivot_row != k {
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
                b.swap(k, pivot_row);
            }
            // Eliminate below the pivot
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / akk;
                a[i * n + k] = factor;
                for j in (k + 1)..n {
                    a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
                }
            }
        }

        // Forward solve Ly = Pb (b is permuted)
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum;
        }
        // Backward solve Ux = y
        for i in (0..n).rev() {
            let mut sum = b[i];
            for k in (i + 1)..n {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "nalgebra"))]
mod tests {
    use crate::linalg::matrix::Matrix;
    use nalgebra::Vector2;

    #[test]
    fn solve_full_2x2() {
        // A = [[3, 2],[1, 4]], b = [5, 6] -> x = [0.8, 1.3]
        let mut a: Matrix<f64> = Matrix::full(2, 2);
        a[(0, 0)] = 3.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 4.0;
        let b = Vector2::new(5.0, 6.0);
        let x = a.lin_solve(b).unwrap();
        // Solve manually: [[3,2],[1,4]] x = [5,6] => x = [ (20-12)/10, (15-5)/10 ] = [0.8, 1.3]
        assert!((x.x - 0.8).abs() < 1e-12);
        assert!((x.y - 1.3).abs() < 1e-12);
    }

    #[test]
    fn solve_sparse_2x2() {
        let a: Matrix<f64> = Matrix::sparse_from_triplets(
            2,
            2,
            vec![(0, 0, 3.0), (0, 1, 2.0), (1, 0, 1.0), (1, 1, 4.0)],
        );
        let b = Vector2::new(5.0, 6.0);
        let x = a.lin_solve(b).unwrap();
        assert!((x.x - 0.8).abs() < 1e-12);
        assert!((x.y - 1.3).abs() < 1e-12);
    }

    #[test]
    fn solve_sparse_mut_2x2() {
        let a: Matrix<f64> = Matrix::sparse_from_triplets(
            2,
            2,
            vec![(0, 0, 3.0), (0, 1, 2.0), (1, 0, 1.0), (1, 1, 4.0)],
        );
        let mut b = [5.0_f64, 6.0];
        a.lin_solve_mut(&mut b).unwrap();
        assert!((b[0] - 0.8).abs() < 1e-12);
        assert!((b[1] - 1.3).abs() < 1e-12);
    }
}
