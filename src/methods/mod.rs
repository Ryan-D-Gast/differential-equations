//! Numerical Methods for Differential Equations

mod h_init;

// --- Explicit Runge-Kutta Methods ---
mod erk;
pub use erk::ExplicitRungeKutta;

// --- Implicit Runge-Kutta Methods ---
mod irk;
pub use irk::ImplicitRungeKutta;

// --- Adams Predictor-Corrector Methods ---
mod apc;
pub use apc::AdamsPredictorCorrector;

// --- Typestate Categories for Differential Equations Types ---
pub struct Ordinary;
pub struct Delay;
pub struct Stochastic;

// --- Typestate Categories for Numerical Methods Families ---

/// Fixed-step methods
pub struct Fixed;

/// Adaptive-step methods
pub struct Adaptive;

/// Explicit Adaptive-step methods by Dormand-Prince
/// 
/// Note that technically, Dormand-Prince is a specific adaptive method, but we keep it as a separate category 
/// because the there are optimizations for the primary stages, error estimation, and dense output interpolation
/// that can not be generalized to all adaptive methods and thus requires it's own category. 
/// Non Dormand-Prince adaptive methods might also be implemented in this category that share the same optimizations.
pub struct DormandPrince;