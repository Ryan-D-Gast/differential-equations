//! Stochastic Differential Equations (SDEs) module.

mod sde;
pub use sde::SDE;

pub mod methods;
pub use methods::{
    EM,
    Milstein,
    RKM4,
};

mod numerical_method;
pub use numerical_method::{
    NumEvals, 
    NumericalMethod
};

mod solve;
pub use solve::{
    solve_sde,
    SDEProblem,
};

// Re-exports to allow users to only import necessary components
pub use crate::shared::*;