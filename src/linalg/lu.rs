//! LU decomposition algorithms.
//!
//! This module provides LU decomposition with partial pivoting for real and complex matrices.
//!
//! # References
//! - Hairer, E., & Wanner, G. (1996). Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems. Springer.

use crate::{
    linalg::{Matrix, error::LinalgError},
    traits::Real,
};

/// LU decomposition with partial pivoting
///
/// This function performs LU decomposition with partial pivoting on a square matrix,
/// factorizing a matrix A into the product PA = LU where:
/// - P is a permutation matrix (represented by pivot indices)
/// - L is unit lower triangular (with implicit unit diagonal)
/// - U is upper triangular
///
/// # Arguments
/// * `a` - Square matrix to decompose (modified in-place to store L and U)
/// * `ip` - Pivot index slice (must have length equal to matrix size)
///
/// # Returns
/// * `Ok(())` - Decomposition successful
/// * `Err(k)` - Matrix is singular, detected at step k (1-indexed)
///
/// # Algorithm
/// The decomposition proceeds in n-1 stages. At each stage k:
/// 1. **Pivoting**: Find the largest element in column k below the diagonal
/// 2. **Row exchange**: Swap rows to bring the pivot to the diagonal
/// 3. **Elimination**: Use the pivot to eliminate elements below it
/// 4. **Update**: Apply the elimination to the remaining submatrix
///
/// The pivot information is stored in `ip` where `ip[k]` contains the row index
/// that was swapped with row k during stage k.
///
/// # Mathematical Background
/// LU decomposition with partial pivoting factors PA = LU where:
/// - The permutation P ensures numerical stability by choosing the largest pivot
/// - L has unit diagonal (1's) and the multipliers below the diagonal
/// - U is upper triangular with the pivots on the diagonal
/// - The factorization satisfies: A = P⁻¹LU
///
/// # Storage
/// After decomposition, the matrix `a` contains:
/// - Upper triangle and diagonal: the U factor
/// - Strict lower triangle: the L factor (without the unit diagonal)
///
/// # Examples
/// ```rust,ignore
/// use differential_equations::linalg::{Matrix, lu::dec};
///
/// let mut a = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 1.0]);
/// let mut ip = [0; 2];
///
/// match dec(&mut a, &mut ip) {
///     Ok(()) => println!("Decomposition successful"),
///     Err(err) => println!("Decomposition failed: {}", err),
/// }
/// ```
///
/// # Errors
/// Returns [`LinalgError`] if the matrix is not square, pivot slice has wrong size, or matrix is singular.
pub fn dec<T: Real>(a: &mut Matrix<T>, ip: &mut [usize]) -> Result<(), LinalgError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::BadInput {
            message: format!("Matrix is not square: {}x{}", n, a.ncols()),
        });
    }

    if ip.len() != n {
        return Err(LinalgError::PivotSizeMismatch {
            expected: n,
            actual: ip.len(),
        });
    }

    if n == 1 {
        if a[(0, 0)] == T::zero() {
            return Err(LinalgError::Singular { step: 1 });
        }
        ip[0] = 0;
        return Ok(());
    }

    let nm1 = n - 1;
    for k in 0..nm1 {
        let kp1 = k + 1;

        // Find pivot - search for largest magnitude element in column k
        let mut m = k;
        let mut max_val = a[(k, k)].abs();
        for i in kp1..n {
            let val = a[(i, k)].abs();
            if val > max_val {
                max_val = val;
                m = i;
            }
        }

        ip[k] = m;
        let pivot = a[(m, k)];

        // Swap rows if needed for pivoting
        if m != k {
            for j in 0..n {
                let temp = a[(m, j)];
                a[(m, j)] = a[(k, j)];
                a[(k, j)] = temp;
            }
        }

        // Check for singularity
        if pivot == T::zero() {
            return Err(LinalgError::Singular { step: k + 1 });
        }

        // Scale column - store negative multipliers
        let t = T::one() / pivot;
        for i in kp1..n {
            a[(i, k)] = -a[(i, k)] * t;
        }

        // Update remaining submatrix using outer product
        for j in kp1..n {
            let ajk = a[(k, j)];

            // Apply row exchange to this column if needed
            if m != k {
                let temp = a[(m, j)];
                a[(m, j)] = a[(k, j)];
                a[(k, j)] = temp;
            }

            // Apply elimination if the pivot column element is non-zero
            if ajk != T::zero() {
                for i in kp1..n {
                    a[(i, j)] = a[(i, j)] + a[(i, k)] * ajk;
                }
            }
        }
    }

    // Check if the final diagonal element is non-zero
    if a[(n - 1, n - 1)] == T::zero() {
        return Err(LinalgError::Singular { step: n });
    }

    Ok(())
}

/// Complex LU decomposition with partial pivoting
///
/// This function performs LU decomposition with partial pivoting on a complex matrix
/// represented by separate real and imaginary parts. It factorizes a complex matrix
/// (AR + i*AI) into the product P(AR + i*AI) = LU where:
/// - P is a permutation matrix (represented by pivot indices)
/// - L is unit lower triangular (with implicit unit diagonal)
/// - U is upper triangular
///
/// # Arguments
/// * `ar` - Real part of the square matrix to decompose (modified in-place)
/// * `ai` - Imaginary part of the square matrix to decompose (modified in-place)
/// * `ip` - Pivot index slice (must have length equal to matrix size)
///
/// # Returns
/// * `Ok(())` - Decomposition successful
/// * `Err(k)` - Matrix is singular, detected at step k (1-indexed)
///
/// # Algorithm
/// Similar to real LU decomposition, but with complex arithmetic:
/// 1. **Pivoting**: Find the largest magnitude complex element in column k
/// 2. **Row exchange**: Swap rows to bring the pivot to the diagonal
/// 3. **Elimination**: Use complex arithmetic to eliminate elements below the pivot
/// 4. **Update**: Apply complex elimination to the remaining submatrix
///
/// The magnitude of a complex number (a + bi) is computed as |a| + |b| for efficiency.
/// All complex operations are performed using separate real and imaginary components.
///
/// # Mathematical Background
/// Complex LU decomposition factors P(A + iB) = LU where the complex arithmetic
/// is handled explicitly:
/// - Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
/// - Complex division: (a + bi)/(c + di) = [(ac + bd) + (bc - ad)i]/(c² + d²)
///
/// # Examples
/// ```rust,ignore
/// use differential_equations::linalg::{Matrix, lu::decc};
///
/// let mut ar = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
/// let mut ai = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
/// let mut ip = [0; 2];
///
/// match decc(&mut ar, &mut ai, &mut ip) {
///     Ok(()) => println!("Complex decomposition successful"),
///     Err(err) => println!("Complex decomposition failed: {}", err),
/// }
/// ```
///
/// # Errors
/// Returns [`LinalgError`] if matrices have inconsistent dimensions, pivot slice has wrong size, or matrix is singular.
pub fn decc<T: Real>(
    ar: &mut Matrix<T>,
    ai: &mut Matrix<T>,
    ip: &mut [usize],
) -> Result<(), LinalgError> {
    let n = ar.nrows();
    if n != ar.ncols() || n != ai.nrows() || n != ai.ncols() {
        return Err(LinalgError::BadInput {
            message: format!(
                "Matrix dimensions inconsistent: {}x{}, {}x{}",
                ar.nrows(),
                ar.ncols(),
                ai.nrows(),
                ai.ncols()
            ),
        });
    }

    if ip.len() != n {
        return Err(LinalgError::PivotSizeMismatch {
            expected: n,
            actual: ip.len(),
        });
    }

    if n == 1 {
        if ar[(0, 0)].abs() + ai[(0, 0)].abs() == T::zero() {
            return Err(LinalgError::Singular { step: 1 });
        }
        ip[0] = 0;
        return Ok(());
    }

    let nm1 = n - 1;
    for k in 0..nm1 {
        let kp1 = k + 1;

        // Find pivot - largest magnitude complex number
        let mut m = k;
        let mut max_val = ar[(k, k)].abs() + ai[(k, k)].abs();
        for i in kp1..n {
            let val = ar[(i, k)].abs() + ai[(i, k)].abs();
            if val > max_val {
                max_val = val;
                m = i;
            }
        }

        ip[k] = m;
        let mut tr = ar[(m, k)];
        let mut ti = ai[(m, k)];

        // Swap rows if needed
        if m != k {
            for j in 0..n {
                let temp_r = ar[(m, j)];
                let temp_i = ai[(m, j)];
                ar[(m, j)] = ar[(k, j)];
                ai[(m, j)] = ai[(k, j)];
                ar[(k, j)] = temp_r;
                ai[(k, j)] = temp_i;
            }
        }

        // Check for singularity
        if tr.abs() + ti.abs() == T::zero() {
            return Err(LinalgError::Singular { step: k + 1 });
        }

        // Complex division: 1/(tr + i*ti) = (tr - i*ti)/(tr^2 + ti^2)
        let den = tr * tr + ti * ti;
        tr = tr / den;
        ti = -ti / den;

        // Scale column - store negative multipliers
        for i in kp1..n {
            let prod_r = ar[(i, k)] * tr - ai[(i, k)] * ti;
            let prod_i = ai[(i, k)] * tr + ar[(i, k)] * ti;
            ar[(i, k)] = -prod_r;
            ai[(i, k)] = -prod_i;
        }

        // Update remaining matrix
        for j in kp1..n {
            let mut ajk_r = ar[(k, j)];
            let mut ajk_i = ai[(k, j)];

            // Swap if needed
            if m != k {
                let temp_r = ar[(m, j)];
                let temp_i = ai[(m, j)];
                ar[(m, j)] = ar[(k, j)];
                ai[(m, j)] = ai[(k, j)];
                ar[(k, j)] = temp_r;
                ai[(k, j)] = temp_i;
                ajk_r = temp_r;
                ajk_i = temp_i;
            }

            if ajk_r.abs() + ajk_i.abs() != T::zero() {
                // Optimized cases for better performance
                if ajk_i == T::zero() {
                    // Real multiplication
                    for i in kp1..n {
                        let prod_r = ar[(i, k)] * ajk_r;
                        let prod_i = ai[(i, k)] * ajk_r;
                        ar[(i, j)] = ar[(i, j)] + prod_r;
                        ai[(i, j)] = ai[(i, j)] + prod_i;
                    }
                } else if ajk_r == T::zero() {
                    // Imaginary multiplication
                    for i in kp1..n {
                        let prod_r = -ai[(i, k)] * ajk_i;
                        let prod_i = ar[(i, k)] * ajk_i;
                        ar[(i, j)] = ar[(i, j)] + prod_r;
                        ai[(i, j)] = ai[(i, j)] + prod_i;
                    }
                } else {
                    // Full complex multiplication
                    for i in kp1..n {
                        let prod_r = ar[(i, k)] * ajk_r - ai[(i, k)] * ajk_i;
                        let prod_i = ai[(i, k)] * ajk_r + ar[(i, k)] * ajk_i;
                        ar[(i, j)] = ar[(i, j)] + prod_r;
                        ai[(i, j)] = ai[(i, j)] + prod_i;
                    }
                }
            }
        }
    }

    // Check final diagonal element
    if ar[(n - 1, n - 1)].abs() + ai[(n - 1, n - 1)].abs() == T::zero() {
        return Err(LinalgError::Singular { step: n });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dec_simple() {
        // Test LU decomposition of a simple 2x2 matrix
        let mut a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 4.0, 3.0]);
        let mut ip = [0; 2];

        let result = dec(&mut a, &mut ip);
        assert!(result.is_ok());

        // The matrix should be factorized in-place
        // We can verify that the diagonal elements are non-zero
        assert!(a[(0, 0)].abs() > 1e-10);
        assert!(a[(1, 1)].abs() > 1e-10);
    }

    #[test]
    fn test_dec_singular() {
        // Test with a singular matrix
        let mut a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 0.0]);
        let mut ip = [0; 2];

        let result = dec(&mut a, &mut ip);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LinalgError::Singular { step: 2 });
    }

    #[test]
    fn test_dec_1x1() {
        // Test with a 1x1 matrix
        let mut a = Matrix::from_vec(1, 1, vec![5.0_f64]);
        let mut ip = [0; 1];

        let result = dec(&mut a, &mut ip);
        assert!(result.is_ok());
        assert_eq!(ip[0], 0);
    }

    #[test]
    fn test_dec_1x1_singular() {
        // Test with a singular 1x1 matrix
        let mut a = Matrix::from_vec(1, 1, vec![0.0_f64]);
        let mut ip = [0; 1];

        let result = dec(&mut a, &mut ip);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LinalgError::Singular { step: 1 });
    }

    #[test]
    fn test_decc_simple() {
        // Test complex LU decomposition of a simple 2x2 matrix
        let mut ar = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
        let mut ai = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let mut ip = [0; 2];

        let result = decc(&mut ar, &mut ai, &mut ip);
        assert!(result.is_ok());

        // Verify that the diagonal elements have non-zero magnitude
        let diag0_mag = ar[(0, 0)].abs() + ai[(0, 0)].abs();
        let diag1_mag = ar[(1, 1)].abs() + ai[(1, 1)].abs();
        assert!(diag0_mag > 1e-10);
        assert!(diag1_mag > 1e-10);
    }

    #[test]
    fn test_decc_singular() {
        // Test with a singular complex matrix
        let mut ar = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 1.0, 1.0]);
        let mut ai = Matrix::from_vec(2, 2, vec![0.0_f64, 0.0, 0.0, 0.0]);
        let mut ip = [0; 2];

        let result = decc(&mut ar, &mut ai, &mut ip);
        assert!(result.is_err());
    }

    #[test]
    fn test_decc_1x1() {
        // Test with a 1x1 complex matrix
        let mut ar = Matrix::from_vec(1, 1, vec![3.0_f64]);
        let mut ai = Matrix::from_vec(1, 1, vec![4.0_f64]); // 3 + 4i
        let mut ip = [0; 1];

        let result = decc(&mut ar, &mut ai, &mut ip);
        assert!(result.is_ok());
        assert_eq!(ip[0], 0);
    }

    #[test]
    fn test_decc_1x1_singular() {
        // Test with a singular 1x1 complex matrix
        let mut ar = Matrix::from_vec(1, 1, vec![0.0_f64]);
        let mut ai = Matrix::from_vec(1, 1, vec![0.0_f64]);
        let mut ip = [0; 1];

        let result = decc(&mut ar, &mut ai, &mut ip);
        assert!(result.is_err());
    }
}
