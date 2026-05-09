//! Linear algebra error types.

/// Errors that can occur during linear algebra operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    /// Input validation error (e.g., non-square matrix, mismatched dimensions)
    BadInput { message: String },
    /// Matrix is singular at the given step (1-indexed)
    Singular { step: usize },
    /// Pivot vector has incorrect size
    PivotSizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::BadInput { message } => {
                write!(f, "Linear algebra input error: {}", message)
            }
            LinalgError::Singular { step } => write!(f, "Matrix is singular at step {}", step),
            LinalgError::PivotSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Pivot vector size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for LinalgError {}
