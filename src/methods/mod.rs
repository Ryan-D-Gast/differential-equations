//! Numerical Methods for Differential Equations

/// Explicit Runge-Kutta Methods
mod erk;
pub use erk::ExplicitRungeKutta;

/// Typestate Categories for Differential Equations Types
pub struct Delay;
pub struct Ordinary;

/// Typestate Categories for Numerical Methods Families
pub struct Classic;