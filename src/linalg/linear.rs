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
/// use differential_equations::linalg::{Matrix, solver::sol};
///
/// // After LU decomposition with partial pivoting
/// let mut b = vec![1.0, 2.0, 3.0];
/// sol(&lu_matrix, &mut b, &pivot_indices);
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
pub fn sol<T: Real, Y: State<T>>(a: &Matrix<T>, b: &mut Y, ip: &[usize]) {
    let n = a.nrows();
    debug_assert_eq!(b.len(), n, "RHS length must match matrix size");

    // Handle trivial case
    if n == 1 {
        let x = b.get(0) / a[(0, 0)];
        b.set(0, x);
        return;
    }

    let nm1 = n - 1;

    // Forward elimination with partial pivoting
    for k in 0..nm1 {
        let kp1 = k + 1;
        let m = ip[k]; // Pivot row index

        // Apply row permutation (swap b[m] and b[k])
        let t = b.get(m);
        let bk = b.get(k);
        b.set(m, bk);
        b.set(k, t);

        // Forward substitution step: b[i] += L[i,k] * b[k]
        for i in kp1..n {
            let bi = b.get(i) + a[(i, k)] * t;
            b.set(i, bi);
        }
    }

    // Back substitution: solve Ux = y
    for kb in 1..n {
        let k = n - kb;

        // Divide by diagonal element
        let xk = b.get(k) / a[(k, k)];
        b.set(k, xk);
        let t = -xk;

        // Back substitution step: b[i] += U[i,k] * (-b[k])
        for i in 0..k {
            let bi = b.get(i) + a[(i, k)] * t;
            b.set(i, bi);
        }
    }

    // Final division for the first element
    let x0 = b.get(0) / a[(0, 0)];
    b.set(0, x0);
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
/// use differential_equations::linalg::{Matrix, solver::solc};
///
/// // After complex LU decomposition
/// let mut br = vec![1.0, 2.0];
/// let mut bi = vec![0.5, 1.5];
/// solc(&ar_matrix, &ai_matrix, &mut br, &mut bi, &pivot_indices);
/// // br, bi now contain the complex solution
/// ```
///
/// # Mathematical Background
/// For a complex system (A + iB)x = (c + id), the LU factorization gives:
/// P(A + iB) = L + iM + N + iO
/// where the solver handles the complex arithmetic explicitly.
pub fn solc<T: Real, Y: State<T>>(
    ar: &Matrix<T>,
    ai: &Matrix<T>,
    br: &mut Y,
    bi: &mut Y,
    ip: &[usize],
) {
    let n = ar.nrows();
    debug_assert_eq!(br.len(), n, "RHS length must match matrix size");
    debug_assert_eq!(bi.len(), n, "RHS length must match matrix size");

    // Handle trivial case with complex division
    if n == 1 {
        // Complex division: (br + i*bi) / (ar + i*ai)
        let den = ar[(0, 0)] * ar[(0, 0)] + ai[(0, 0)] * ai[(0, 0)];
        let temp_r = (br.get(0) * ar[(0, 0)] + bi.get(0) * ai[(0, 0)]) / den;
        let temp_i = (bi.get(0) * ar[(0, 0)] - br.get(0) * ai[(0, 0)]) / den;
        br.set(0, temp_r);
        bi.set(0, temp_i);
        return;
    }

    let nm1 = n - 1;

    // Forward elimination with partial pivoting (complex version)
    for k in 0..nm1 {
        let kp1 = k + 1;
        let m = ip[k]; // Pivot row index

        // Apply row permutation to both real and imaginary parts
        let tr = br.get(m);
        let ti = bi.get(m);
        let brk = br.get(k);
        let bik = bi.get(k);
        br.set(m, brk);
        bi.set(m, bik);
        br.set(k, tr);
        bi.set(k, ti);

        // Complex forward substitution: b[i] += L[i,k] * t
        for i in kp1..n {
            // Complex multiplication: (ar[i,k] + i*ai[i,k]) * (tr + i*ti)
            let prod_r = ar[(i, k)] * tr - ai[(i, k)] * ti;
            let prod_i = ai[(i, k)] * tr + ar[(i, k)] * ti;
            let bir = br.get(i) + prod_r;
            let bii = bi.get(i) + prod_i;
            br.set(i, bir);
            bi.set(i, bii);
        }
    }

    // Complex back substitution
    for kb in 1..n {
        let k = n - kb;

        // Complex division: b[k] = b[k] / a[k,k]
        let den = ar[(k, k)] * ar[(k, k)] + ai[(k, k)] * ai[(k, k)];
        let temp_r = (br.get(k) * ar[(k, k)] + bi.get(k) * ai[(k, k)]) / den;
        let temp_i = (bi.get(k) * ar[(k, k)] - br.get(k) * ai[(k, k)]) / den;
        br.set(k, temp_r);
        bi.set(k, temp_i);

        // Prepare for back substitution: t = -b[k]
        let tr = -br.get(k);
        let ti = -bi.get(k);

        // Complex back substitution step: b[i] += a[i,k] * t
        for i in 0..k {
            let prod_r = ar[(i, k)] * tr - ai[(i, k)] * ti;
            let prod_i = ai[(i, k)] * tr + ar[(i, k)] * ti;
            let bir = br.get(i) + prod_r;
            let bii = bi.get(i) + prod_i;
            br.set(i, bir);
            bi.set(i, bii);
        }
    }

    // Final complex division for the first element
    let den = ar[(0, 0)] * ar[(0, 0)] + ai[(0, 0)] * ai[(0, 0)];
    let temp_r = (br.get(0) * ar[(0, 0)] + bi.get(0) * ai[(0, 0)]) / den;
    let temp_i = (bi.get(0) * ar[(0, 0)] - br.get(0) * ai[(0, 0)]) / den;
    br.set(0, temp_r);
    bi.set(0, temp_i);
}

#[cfg(test)]
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

        sol(&a, &mut b, &ip);

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

        solc(&ar, &ai, &mut br, &mut bi, &ip);

        // Solution (s,0)hould be (2 + 0i) / (1 + i) = (1 - i)
        assert!((br[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((bi[0] - (-1.0)).abs() < 1e-10);
    }
}
