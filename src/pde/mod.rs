//! Partial Differential Equations (PDE) module.
//!
//! The initial PDE support uses the method of lines: discretize space first, then
//! solve the resulting ODE system with the existing IVP solvers.

mod boundary;
mod grid;
mod method_of_lines;
mod navier_stokes;
mod pde;
mod semi_discrete;
mod spatial_discretization;

pub use boundary::{BoundaryCondition, BoundaryConditions};
pub use grid::{BoundaryFace, Side, StructuredGrid};
pub use method_of_lines::MethodOfLines;
pub use navier_stokes::{ProjectionMethod, ProjectionSemiDiscrete};
pub use pde::PDE;
pub use semi_discrete::{SemiDiscretePde, SpatialScheme};
pub use spatial_discretization::SpatialDiscretization;
