//! Linear algebra types and utilities.

mod schur;
mod util;

pub mod matrix;

pub use matrix::{Matrix, MatrixStorage};
pub use schur::schur_complement;
pub use util::*;
