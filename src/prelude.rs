#![allow(clippy::all)]
//! # Library Prelude
//!
//! Convenient import for all core types, traits, and solvers in the `differential-equations` crate.
//!
//! ## What is re-exported?
//!
//! - All major solver types for ODE, DDE, and SDE problems
//! - The IVP builder and equation traits for each equation class
//! - Output control, event, and solution types
//! - Common utility types (e.g., error, status, interpolation)
//! - Key nalgebra types for vector/matrix work
//!
//! ## Solver Naming
//!
//! Solvers are accessed as `MethodType::solver_name`, e.g. `ExplicitRungeKutta::dop853`, `ImplicitRungeKutta::radau5`.
//! This keeps the API organized and avoids naming conflicts.
//!
//! ## Example Solvers
//!
//! - `ExplicitRungeKutta::dop853`, `dopri5`, `euler`, `rk4`, `rkv65e`, `rkv98e`
//! - `ImplicitRungeKutta::radau5`
//! - `DiagonallyImplicitRungeKutta::kvaerno745`
//!
//! ## Method Support Table
//!
//! | Method                        | ODE | DDE | SDE | DAE |
//! |-------------------------------|:---:|:---:|:---:|:---:|
//! | ExplicitRungeKutta            |  X  |  X  | (X) |     |
//! | ImplicitRungeKutta            |  X  |     |     | {X} |
//! | DiagonallyImplicitRungeKutta  |  X  |     |     |     |
//! | AdamsPredictorCorrector       |  X  |     |     |     |
//!
//! - `X` = Supported
//! - `(X)` = Supported for fixed step only (e.g., Euler, RK4)
//! - `{X}` = Supported using ImplicitRungeKutta::radau5 solver only
//!
//! For full examples and advanced usage, see the [examples directory](https://github.com/Ryan-D-Gast/differential-equations/tree/master/examples).
//!

// Numerical Methods
pub use crate::methods::{
    AdamsPredictorCorrector, BDF, DiagonallyImplicitRungeKutta, ExplicitRungeKutta,
    ImplicitRungeKutta,
};

// Initial Value Problems
pub use crate::ivp::IVP;

// Equation Traits
pub use crate::dae::DAE;
pub use crate::dde::DDE;
pub use crate::ode::ODE;
pub use crate::sde::SDE;

// Output, Events, and Solution Types
pub use crate::control::ControlFlag;
pub use crate::derive::State;
pub use crate::error::Error;
pub use crate::interpolate::Interpolation;
pub use crate::solout::{CrossingDirection, Event, EventConfig, Solout};
pub use crate::solution::Solution;
pub use crate::stats::Evals;
pub use crate::status::Status;

// Linear Algebra
pub use crate::linalg::Matrix;
