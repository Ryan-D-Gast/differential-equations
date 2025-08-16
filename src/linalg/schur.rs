//! Schur complement helpers for block systems used in IRK solvers.

use crate::traits::{Real, State};

use super::SquareMatrix;

/// Solve the 2x2 block system using the (explicit) Schur complement:
/// [A B; C D] [x;y] = [r;s]
/// Returns (x, y).
///
/// Notes:
/// - This forms the dense Schur complement S = D - C A^{-1} B explicitly.
///   For small per-stage blocks (common in IRK), this is acceptable and simple.
/// - For larger blocks, prefer an operator-based approach that applies S without forming it.
pub fn schur_complement<T: Real, V: State<T>>(
    a: &SquareMatrix<T>,
    b: &SquareMatrix<T>,
    c: &SquareMatrix<T>,
    d: &SquareMatrix<T>,
    r: V,
    s: V,
) -> (V, V) {
    let n = a.n();
    assert_eq!(b.n(), n, "block size mismatch: B");
    assert_eq!(c.n(), n, "block size mismatch: C");
    assert_eq!(d.n(), n, "block size mismatch: D");
    assert_eq!(r.len(), n, "rhs r size mismatch");
    assert_eq!(s.len(), n, "rhs s size mismatch");

    // Helper: solve with A and D using existing dense LU path
    let solve_a = |rhs: V| a.lin_solve(rhs);

    // Build dense Schur complement S = D - C A^{-1} B, as a dense Full matrix
    // We'll fill column-by-column using basis vectors e_j.
    let mut s_dense = SquareMatrix::zeros(n);
    for j in 0..n {
        // e_j
        let mut e = V::zeros();
        e.set(j, T::one());
        // u = B e_j
        let u = b.mul_state(&e);
        // v = A^{-1} u
        let v = solve_a(u);
        // z = C v
        let z = c.mul_state(&v);
        // column j of S is (D e_j - z)
        let d_ej = d.mul_state(&{
            let mut e2 = V::zeros();
            e2.set(j, T::one());
            e2
        });
        for i in 0..n {
            let val = d_ej.get(i) - z.get(i);
            s_dense[(i, j)] = val;
        }
    }

    // Compute w = s - C A^{-1} r
    let ar = solve_a(r);
    let car = c.mul_state(&ar);
    let mut w = V::zeros();
    for i in 0..n {
        w.set(i, s.get(i) - car.get(i));
    }

    // Solve S y = w
    let y = s_dense.lin_solve(w);

    // Back-substitute for x: A x = r - B y
    let by = b.mul_state(&y);
    let mut rhs_x = V::zeros();
    for i in 0..n {
        rhs_x.set(i, r.get(i) - by.get(i));
    }
    let x = solve_a(rhs_x);

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::{SquareMatrix, schur_complement};
    use nalgebra::Vector2;

    fn approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{} != {}", a, b);
    }

    #[test]
    fn schur_trivial_identity_blocks() {
        let a: SquareMatrix<f64> = SquareMatrix::identity(2);
        let d: SquareMatrix<f64> = SquareMatrix::identity(2);
        let b: SquareMatrix<f64> = SquareMatrix::zeros(2);
        let c: SquareMatrix<f64> = SquareMatrix::zeros(2);

        let x_true = Vector2::new(1.0, -2.0);
        let y_true = Vector2::new(3.0, 4.0);

        // r = A x + B y = x; s = C x + D y = y
        let r = a.mul_state(&x_true);
        let s = d.mul_state(&y_true);

        let (x, y) = schur_complement(&a, &b, &c, &d, r, s);
        approx_eq(x.x, x_true.x);
        approx_eq(x.y, x_true.y);
        approx_eq(y.x, y_true.x);
        approx_eq(y.y, y_true.y);
    }

    #[test]
    fn schur_mixed_blocks_small_dense() {
        // Choose small invertible A and D, and simple B, C
        let a: SquareMatrix<f64> = SquareMatrix::full(2, vec![3.0, 1.0, 2.0, 4.0]);
        let d: SquareMatrix<f64> = SquareMatrix::full(2, vec![2.0, 0.5, 1.0, 3.0]);
        let b: SquareMatrix<f64> = SquareMatrix::full(2, vec![1.0, 0.0, 0.0, 1.0]); // I
        let c: SquareMatrix<f64> = SquareMatrix::full(2, vec![0.5, 0.0, 0.0, 0.5]); // 0.5 I

        let x_true = Vector2::new(1.0, -2.0);
        let y_true = Vector2::new(3.0, 4.0);

        // r = A x + B y, s = C x + D y
        let r = {
            let ax = a.mul_state(&x_true);
            let by = b.mul_state(&y_true);
            Vector2::new(ax.x + by.x, ax.y + by.y)
        };
        let s = {
            let cx = c.mul_state(&x_true);
            let dy = d.mul_state(&y_true);
            Vector2::new(cx.x + dy.x, cx.y + dy.y)
        };

        let (x, y) = schur_complement(&a, &b, &c, &d, r, s);
        approx_eq(x.x, x_true.x);
        approx_eq(x.y, x_true.y);
        approx_eq(y.x, y_true.x);
        approx_eq(y.y, y_true.y);
    }
}
