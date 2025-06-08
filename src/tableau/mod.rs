#![allow(dead_code)]
//! Butcher Tableau

// Classic Runge-Kutta methods
mod rk4;

// Embedded Runge-Kutta methods
mod bogacki_shampine;

// Verner's Runge-Kutta methods, source: https://www.sfu.ca/~jverner/
mod rkv76;
mod rkv87;
mod rkv98;

// Dormand-Prince methods
mod dopri5;
mod dop853;

// Implicit Runge-Kutta methods
mod lobatto;
mod gauss_legendre;
mod radau;

/// Butcher Tableau structure for Runge-Kutta methods.
/// 
/// A Butcher tableau encodes the coefficients of a Runge-Kutta method and provides the
/// necessary components for solving ordinary differential equations. This implementation
/// includes support for embedded methods (for error estimation) and dense output through interpolation.
/// 
/// # Generic Parameters
/// - `S`: Number of stages in the method.
/// - `T`: Primary Stages plus extra stages for interpolation (default is equal to `S`).
/// 
/// # Fields
/// - `c`: Node coefficients (time steps within the interval).
/// - `a`: Runge-Kutta matrix coefficients (coupling between stages).
/// - `b`: Weight coefficients for the primary method's final stage.
/// - `bh`: Weight coefficients for the embedded method (used for error estimation).
///   
///   These allow approximation at any point within the integration step.
/// 
pub struct ButcherTableau<const S: usize, const T: usize = S> {
    pub c: [f64; T],
    pub a: [[f64; T]; T],
    pub b: [f64; S],
    pub bh: Option<[f64; S]>,
    pub bi: Option<[[f64; T]; T]>,
}

impl<const S: usize, const T: usize> ButcherTableau<S, T> {
    /// Number of stages in the method
    pub const STAGES: usize = S;

    /// Number of extra stages for interpolation
    pub const EXTRA_STAGES: usize = T - S;
}

pub trait NumericalMethod<const S: usize> {
    /// Get the Butcher tableau for the method.
    fn tableau(&self) -> &ButcherTableau<S>;

    /// Prepare continuous output coefficients for interpolation.
    fn prepare_continuous_output(&self);
}