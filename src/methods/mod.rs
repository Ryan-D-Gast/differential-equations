//! Numerical Methods for Differential Equations

mod h_init;

// --- Explicit Runge-Kutta Methods ---
mod erk;
pub use erk::ExplicitRungeKutta;

// --- Implicit Runge-Kutta Methods ---
mod irk;
pub use irk::ImplicitRungeKutta;

// --- Diagonally Implicit Runge-Kutta Methods ---
mod dirk;
pub use dirk::DiagonallyImplicitRungeKutta;

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
pub struct DormandPrince;
