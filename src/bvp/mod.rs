//! Boundary Value Problem (BVP) module.
//!
//! Provides traits and types for defining and solving BVP problems.

pub mod bvp;
pub mod problem;
pub mod shooting;
pub mod solve;

pub use bvp::BVP;
pub use problem::BvpProblem;
pub use shooting::ShootingMethod;
pub use solve::BVPMethod;
