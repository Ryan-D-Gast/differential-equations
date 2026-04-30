//! Linear algebra types and utilities.

pub mod error;
mod schur;
mod util;

pub mod linear;
pub mod lu;
pub mod matrix;

pub use error::LinalgError;
pub use linear::{lin_solve, lin_solve_complex};
pub use lu::{lu_decomp, lu_decomp_complex};
pub use matrix::{Matrix, MatrixStorage};
pub use schur::schur_complement;
pub use util::*;
