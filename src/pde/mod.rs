//! Partial Differential Equations (PDE) module.
//!
//! The initial PDE support uses the method of lines: discretize space first, then
//! solve the resulting ODE system with the existing IVP solvers.

mod boundary;
mod grid;
mod maxwell;
mod method_of_lines;
mod pde;
mod semi_discrete;
mod spatial_discretization;

pub use boundary::{
    BoundaryCondition, BoundaryConditions, BoundaryConditionsBuilder,
    BoundaryConditionsBuilderError,
};
pub use grid::{BoundaryFace, Side, StructuredGrid};
pub use maxwell::{SemiDiscreteYee, YeeGrid};
pub use method_of_lines::MethodOfLines;
pub use pde::PDE;
pub use semi_discrete::{SemiDiscretePde, SpatialScheme};
pub use spatial_discretization::SpatialDiscretization;

pub mod finite_volume;
pub use finite_volume::{FiniteVolume, Limiter, NumericalFlux, Reconstruction};
