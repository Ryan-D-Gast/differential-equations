//! Delay Differential Equations (DDE) Module

mod dde_problem;
pub use dde_problem::DDEProblem;

mod solve_dde;
pub use solve_dde::solve_dde;

mod dde;
pub use dde::DDE;

mod numerical_method;
pub use numerical_method::NumericalMethod;

pub mod methods;