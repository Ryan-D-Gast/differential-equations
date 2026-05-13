//! Boundary Value Problem (BVP) module.
//!
//! Provides traits and types for defining and solving BVP problems.

pub mod bvp;

pub use bvp::BVP;
pub use crate::methods::bvp::{BVPMethod, ShootingMethod};
