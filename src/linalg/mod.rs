//! Linear algebra types and utilities.

pub mod error;
mod schur;
mod util;

pub mod linear;
pub mod lu;
pub mod matrix;

pub use error::LinalgError;
pub use linear::{sol, solc};
pub use lu::{dec, decc};
pub use matrix::{Matrix, MatrixStorage};
pub use schur::schur_complement;
pub use util::*;
