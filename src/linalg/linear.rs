//! Linear system solvers for LU-decomposed matrices.
//!
//! This module provides solvers for linear systems that have already been LU-decomposed
//! with partial pivoting.
//!
//! # References
//! - Hairer, E., & Wanner, G. (1996). Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems. Springer.

use crate::{
    linalg::Matrix,
    traits::{Real, State},
};

/// Solves a real linear system A*X = B using LU decomposition with partial pivoting.
///
/// The matrix `a` must have been previously decomposed using LU factorization with
/// partial pivoting, and `ip` contains the pivot indices from that decomposition.
///
/// # Arguments
/// * `a` - The LU-decomposed matrix (L and U stored in the same matrix)
/// * `b` - On input: right-hand side vector; On output: solution vector X
/// * `ip` - Pivot indices from the LU decomposition
///
/// # Algorithm
/// The solver performs two phases:
/// 1. **Forward elimination**: Applies the row permutations and solves Ly = Pb
/// 2. **Back substitution**: Solves Ux = y
///
/// Where P is the permutation matrix encoded by the pivot vector `ip`.
///
/// # Panics
/// Panics if the matrix dimensions are inconsistent or if indices are out of bounds.
///
/// # Examples
/// ```rust,ignore
/// use differential_equations::linalg::{Matrix, solver::lin_solve};
///
/// // After LU decomposition with partial pivoting
/// let mut b = vec![1.0, 2.0, 3.0];
/// lin_solve(&lu_matrix, &mut b, &pivot_indices);
/// // b now contains the solution
/// ```
///
/// # Mathematical Background
/// For a system Ax = b where A has been factorized as PA = LU:
/// - P is a permutation matrix (represented by pivot indices)
/// - L is unit lower triangular
/// - U is upper triangular
///
/// The solution process is:
/// 1. Solve Ly = Pb (forward substitution with pivoting)
/// 2. Solve Ux = y (back substitution)
use crate::linalg::MatrixStorage;

pub fn lin_solve<T: Real, Y: State<T>>(a: &Matrix<T>, b: &mut Y, ip: &[usize]) {
    if let MatrixStorage::Sparse { ref coords, .. } = a.storage {
        // Fast path for sparse LU solve if matrix is stored sparsely
        let n = a.nrows();

        // 1. Convert coords to row-based representation for faster access
        let mut rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for &(r, c, v) in coords.iter() {
            rows[r].push((c, v));
        }
        for row in rows.iter_mut() {
            row.sort_by_key(|&(c, _)| c);
        }

        // 2. Forward elimination Ly = Pb
        for k in 0..n - 1 {
            let m = ip[k];
            let tk = b.get_component(k);
            let tm = b.get_component(m);
            b.set_component(k, tm);
            b.set_component(m, tk);

            let pivot_val = b.get_component(k);

            for i in k + 1..n {
                if let Ok(pos) = rows[i].binary_search_by_key(&k, |&(c, _)| c) {
                    let val = rows[i][pos].1;
                    let current = b.get_component(i);
                    b.set_component(i, current - val * pivot_val);
                }
            }
        }

        // 3. Back substitution Ux = y
        for i in (0..n).rev() {
            let mut sum = b.get_component(i);
            let mut diag = T::one();

            for &(col, val) in rows[i].iter() {
                if col > i {
                    sum -= val * b.get_component(col);
                } else if col == i {
                    diag = val;
                }
            }

            b.set_component(i, sum / diag);
        }
    } else {
        b.apply_linear_solve(&a.data, ip);
    }
}

/// Solves a complex linear system (AR + i*AI)*X = (BR + i*BI) using LU decomposition.
///
/// The complex matrix (AR + i*AI) must have been previously decomposed using complex
/// LU factorization with partial pivoting, and `ip` contains the pivot indices.
///
/// # Arguments
/// * `ar` - Real part of the LU-decomposed complex matrix
/// * `ai` - Imaginary part of the LU-decomposed complex matrix  
/// * `br` - On input: real part of RHS; On output: real part of solution
/// * `bi` - On input: imaginary part of RHS; On output: imaginary part of solution
/// * `ip` - Pivot indices from the complex LU decomposition
///
/// # Algorithm
/// Similar to the real case, but with complex arithmetic:
/// 1. **Forward elimination**: Applies permutations and complex forward substitution
/// 2. **Back substitution**: Complex back substitution with division
///
/// All complex operations are performed using separate real and imaginary parts:
/// - Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
/// - Complex division: (a + bi)/(c + di) = [(ac + bd) + (bc - ad)i]/(c² + d²)
///
/// # Panics
/// Panics if matrix dimensions are inconsistent or if indices are out of bounds.
///
/// # Examples
/// ```rust,ignore
/// use differential_equations::linalg::{Matrix, lin_solve_complex};
///
/// // After complex LU decomposition
/// let mut br = vec![1.0, 2.0];
/// let mut bi = vec![0.5, 1.5];
/// lin_solve_complex(&ar_matrix, &ai_matrix, &mut br, &mut bi, &pivot_indices);
/// // br, bi now contain the complex solution
/// ```
///
/// # Mathematical Background
/// For a complex system (A + iB)x = (c + id), the LU factorization gives:
/// P(A + iB) = L + iM + N + iO
/// where the solver handles the complex arithmetic explicitly.
pub fn lin_solve_complex<T: Real, Y: State<T>>(
    ar: &Matrix<T>,
    ai: &Matrix<T>,
    br: &mut Y,
    bi: &mut Y,
    ip: &[usize],
) {
    br.apply_complex_linear_solve(bi, &ar.data, &ai.data, ip);
}

#[cfg(all(test, feature = "nalgebra"))]
mod tests {
    use super::*;
    use nalgebra::SMatrix;

    #[test]
    fn test_sol_simple() {
        // Test a simple case: solve a 1x1 system
        let mut a = Matrix::zeros(1, 1);
        a[(0, 0)] = 2.0;
        let mut b = SMatrix::<f64, 1, 1>::from_element(4.0);
        let ip = vec![0];

        lin_solve(&a, &mut b, &ip);

        // Solution( ,0)should be 4.0 / 2.0 = 2.0
        assert!((b[0] - 2.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_solc_simple() {
        // Test with a simple complex system
        let mut ar = Matrix::zeros(1, 1);
        let mut ai = Matrix::zeros(1, 1);
        ar[(0, 0)] = 1.0;
        ai[(0, 0)] = 1.0; // Matrix element is (1 + i)
        let mut br = SMatrix::<f64, 1, 1>::from_element(2.0);
        let mut bi = SMatrix::<f64, 1, 1>::from_element(0.0); // RHS is (2 + 0i)
        let ip = vec![0];

        lin_solve_complex(&ar, &ai, &mut br, &mut bi, &ip);

        // Solution (s,0)hould be (2 + 0i) / (1 + i) = (1 - i)
        assert!((br[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((bi[0] - (-1.0)).abs() < 1e-10);
    }
}
