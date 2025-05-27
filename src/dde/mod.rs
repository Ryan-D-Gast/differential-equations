//! Delay Differential Equations (DDE) Module

mod problem;
pub use problem::DDEProblem;

mod solve;
pub use solve::solve_dde;

mod dde;
pub use dde::DDE;

mod numerical_method;
pub use numerical_method::DDENumericalMethod;

pub mod methods;
