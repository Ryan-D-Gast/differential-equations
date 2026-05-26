//! Incompressible Navier-Stokes backend.
//!
//! This module provides a fractional-step (projection) method backend for
//! solving incompressible Navier-Stokes equations.

mod projection;

pub use projection::ProjectionMethod;
