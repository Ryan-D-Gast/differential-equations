//! NumericalMethods for Ordinary Differential Equations (ODEs).

// Adams Methods for solving ordinary differential equations.
pub mod adams;

// Initial step size determination methods for ODEs.
mod h_init;
pub use h_init::h_init;
