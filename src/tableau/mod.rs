#![allow(dead_code)]
//! Butcher Tableau

// Verner's Runge-Kutta methods, source: https://www.sfu.ca/~jverner/
mod rkv76;
mod rkv87;
mod rkv98;

// Dormand-Prince methods
mod dopri5;
mod dop853;

/// Butcher Tableau structure for Runge-Kutta methods.
/// 
/// A Butcher tableau encodes the coefficients of a Runge-Kutta method and provides the
/// necessary components for solving ordinary differential equations. This implementation
/// includes support for embedded methods (for error estimation) and dense output through interpolation.
/// 
/// # Generic Parameters
/// - `O`: Order of accuracy of the primary method.
/// - `I`: Order of accuracy of the interpolation scheme.
/// - `W`: Number of weights used for interpolation.
/// - `D`: Number of additional stages used for dense output interpolation.
/// - `S`: Number of stages in the primary method.
/// - `T`: Total number of stages including extra stages for interpolation.
///   
///   Note: Once const generic calculations are stable, this can be replaced with just
///   the number of extra stages, as T = S + extra_stages.
/// 
/// # Fields
/// - `c`: Node coefficients (time steps within the interval).
/// - `a`: Runge-Kutta matrix coefficients (coupling between stages).
/// - `b`: Weight coefficients for the primary method's final stage.
/// - `bh`: Weight coefficients for the embedded method (used for error estimation).
/// - `bi`: Interpolation coefficients for dense output calculations.
///   
///   These allow approximation at any point within the integration step.
/// 
pub struct ButcherTableau<const O: usize, const I: usize, const W: usize, const D: usize, const S: usize, const T: usize> {
    pub method: NumicalMethodType,
    pub c: [f64; T],
    pub a: [[f64; T]; T],
    pub b: [f64; S],
    pub bh: Option<[f64; S]>,
    pub bi: Option<[[f64; W]; D]>,
}

pub enum NumicalMethodType {
    /// Explicit Runge-Kutta method
    ExplicitRungeKutta,
    /// Explicit Runge-Kutta method via Dormand-Prince
    DormandPrince,
    /// Implicit Runge-Kutta method
    ImplicitRungeKutta,
}

impl<const O: usize, const I: usize, const W: usize, const D: usize, const S: usize, const T: usize> ButcherTableau<O, I, W, D, S, T> {
    /// Order of accuracy of the primary method
    pub const ORDER: usize = O;
    
    /// Order of accuracy of the interpolation scheme
    pub const INTERPOLATION_ORDER: usize = I;

    /// Number of weights used for interpolation
    pub const WEIGHTS: usize = W;

    /// Number of additional stages used for dense output interpolation
    pub const DENSE_OUTPUT_STAGES: usize = D;
    
    /// Number of stages in the primary method
    pub const STAGES: usize = S;
    
    /// Number of additional stages used for interpolation
    pub const EXTRA_STAGES: usize = T - S;
    
    /// Total number of stages (primary + interpolation)
    pub const TOTAL_STAGES: usize = T;
}

