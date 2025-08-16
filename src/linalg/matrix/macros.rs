//! Convenience macros for constructing matrices.
//!
//! - `matrix![ [a, b], [c, d] ]` or `matrix![ a, b; c, d ]` constructs a full dense matrix (row-major)
//! - `banded_matrix!( k => [vals...], k2 => [vals...] )` constructs a banded matrix by diagonals, inferring size
//!   from the provided diagonal lengths. Here k is the diagonal offset (i - j):
//!   0 is the main diagonal, 1 is the first subdiagonal, -1 is the first superdiagonal, etc.
//!
//! Examples:
//! let m: SquareMatrix<f64> = matrix![ [1.0, 2.0], [3.0, 4.0] ];
//! let b: SquareMatrix<f64> = banded_matrix!(3, 1, 1; (0,0,1.0), (1,0,2.0), (0,1,3.0));

// no local imports required; macros reference items via $crate paths

/// Create a full dense matrix from rows.
/// Usage:
/// - matrix![ [a, b, c], [d, e, f], [g, h, i] ]
/// - matrix![ a, b, c; d, e, f; g, h, i ]
#[macro_export]
macro_rules! matrix {
    // Semicolon-separated rows form: matrix![ a, b; c, d ]
    ( $( $( $x:expr ),+ ) ;+ $(;)? ) => {{
        let rows_vec = vec![ $( vec![ $( $x ),+ ] ),+ ];
        let n = rows_vec.len();
        assert!(rows_vec.iter().all(|r| r.len() == n), "matrix! requires a square n x n list of rows");
        let mut data = Vec::with_capacity(n*n);
        for r in rows_vec.into_iter() { data.extend(r.into_iter()); }
        $crate::linalg::matrix::SquareMatrix::full(n, data)
    }};
    ( $( [ $( $x:expr ),* $(,)? ] ),+ $(,)? ) => {{
        // Collect rows into a Vec<Vec<_>> first
        let rows_vec = vec![ $( vec![ $( $x ),* ] ),+ ];
        let n = rows_vec.len();
        // Ensure square
        assert!(rows_vec.iter().all(|r| r.len() == n), "matrix! requires a square n x n list of rows");
        let mut data = Vec::with_capacity(n*n);
        for r in rows_vec.into_iter() { data.extend(r.into_iter()); }
        $crate::linalg::matrix::SquareMatrix::full(n, data)
    }};
}

/// Create a banded matrix by specifying diagonals. Size and bands are inferred.
/// Usage: banded_matrix!( 0 => [d0...], 1 => [d1...], -1 => [u1...], k => [..], ... )
#[macro_export]
macro_rules! banded_matrix {
    ( $( $k:expr => [ $( $v:expr ),* $(,)? ] ),+ $(,)? ) => {{
        // First pass: determine n, ml, mu from provided diagonals
        let mut n: usize = 0usize;
        let mut ml: usize = 0usize;
        let mut mu: usize = 0usize;
        $( {
            let k: isize = $k as isize;
            let vals = [ $( $v ),* ];
            let len = vals.len();
            let kk: usize = if k < 0 { (-k) as usize } else { k as usize };
            let candidate = len + kk;
            if candidate > n { n = candidate; }
            if k < 0 { if kk > mu { mu = kk; } } else { if kk > ml { ml = kk; } }
        } )+;
        let mut m = $crate::linalg::matrix::SquareMatrix::banded(n, ml, mu);
        // Second pass: fill values; allow shorter diagonals (len <= n - |k|)
        $( {
            let k: isize = $k as isize;
            let vals = [ $( $v ),* ];
            let len = vals.len();
            if k >= 0 {
                let kk = k as usize;
                assert!(len <= n - kk, "diagonal length {} too long for offset {} with inferred n={}", len, k, n);
                for t in 0..len {
                    let j = t;
                    let i = t + kk;
                    m[(i, j)] = vals[t];
                }
            } else {
                let kk = (-k) as usize;
                assert!(len <= n - kk, "diagonal length {} too long for offset {} with inferred n={}", len, k, n);
                for t in 0..len {
                    let i = t;
                    let j = t + kk;
                    m[(i, j)] = vals[t];
                }
            }
        } )+;
        m
    }};
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrix::SquareMatrix;

    #[test]
    fn macro_full_matrix() {
        let m: SquareMatrix<f64> = matrix![ 1.0, 2.0; 3.0, 4.0 ];
        match m {
            SquareMatrix::Full { n, data } => {
                assert_eq!(n, 2);
                assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            _ => panic!("expected full"),
        }
    }

    #[test]
    fn macro_banded_matrix() {
        // Inferred n=3 from main diag len=3
        let b: SquareMatrix<f64> =
            banded_matrix!( 0 => [1.0,1.0,1.0], 1 => [2.0,2.0], -1 => [3.0,3.0] );
        // verify some values
        assert_eq!(b[(0, 0)], 1.0);
        assert_eq!(b[(1, 1)], 1.0);
        assert_eq!(b[(2, 2)], 1.0);
        assert_eq!(b[(1, 0)], 2.0);
        assert_eq!(b[(2, 1)], 2.0);
        assert_eq!(b[(0, 1)], 3.0);
        assert_eq!(b[(1, 2)], 3.0);
        // out of band read yields zero
        assert_eq!(b[(0, 2)], 0.0);
    }

    #[test]
    fn macro_banded_by_diagonals() {
        // 4x4: main diag 1s, first upper 2s, first lower 3s
        let b: SquareMatrix<f64> =
            banded_matrix!( 0 => [1.0,1.0,1.0,1.0], 1 => [2.0,2.0,2.0], -1 => [3.0,3.0,3.0] );
        assert_eq!(b[(0, 0)], 1.0);
        assert_eq!(b[(1, 1)], 1.0);
        assert_eq!(b[(2, 2)], 1.0);
        assert_eq!(b[(3, 3)], 1.0);
        assert_eq!(b[(1, 0)], 2.0);
        assert_eq!(b[(2, 1)], 2.0);
        assert_eq!(b[(3, 2)], 2.0);
        assert_eq!(b[(0, 1)], 3.0);
        assert_eq!(b[(1, 2)], 3.0);
        assert_eq!(b[(2, 3)], 3.0);
        assert_eq!(b[(0, 3)], 0.0);
    }
}
