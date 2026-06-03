//! Partial Differential Equations (PDE) module.
//!
//! The initial PDE support uses the method of lines: discretize space first, then
//! solve the resulting ODE system with the existing IVP solvers.

mod boundary;
mod grid;
mod method_of_lines;
mod pde;
mod projection;
mod semi_discrete;
mod spatial_discretization;
mod yee;

pub use boundary::{
    BoundaryCondition, BoundaryConditions, BoundaryConditionsBuilder,
    BoundaryConditionsBuilderError,
};
pub use grid::{BoundaryFace, Side, StructuredGrid};
pub use method_of_lines::MethodOfLines;
pub use pde::{PDE, PdeFnWrapper, ZeroSource, pde_from_fn, pde_from_fn_flux};
pub use projection::{ProjectionMethod, ProjectionSemiDiscrete};
pub use semi_discrete::{SemiDiscretePde, SpatialScheme};
pub use spatial_discretization::SpatialDiscretization;
pub use yee::{SemiDiscreteYee, YeeGrid, YeeLayout};

pub mod finite_volume;
pub use finite_volume::{FiniteVolume, Limiter, NumericalFlux, Reconstruction};
