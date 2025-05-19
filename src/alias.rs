//! Type Alias for readablity

/// Number of evaluations per step
/// 
/// # Fields
/// * `fcn` - Number of function evaluations
/// * `jac` - Number of Jacobian evaluations
/// 
pub struct Evals {
    pub fcn: usize,
    pub jac: usize,
}

impl Evals {
    /// Create a new Evals struct
    /// 
    /// # Arguments
    /// * `diff` - Number of differntial equation function evaluations
    /// * `jac`  - Number of Jacobian evaluations
    ///
    pub fn new() -> Self {
        Self {
            fcn: 0,
            jac: 0,
        }
    }
}