#![allow(dead_code)]
//! Butcher Tableau

use crate::traits::Real;

// Explicit Runge-Kutta methods
mod runge_kutta;
mod verner;
mod dorman_prince;

// Implicit Runge-Kutta methods
mod lobatto;
mod gauss_legendre;
mod radau;

// Diagonally Implicit Runge-Kutta methods
pub mod dirk;
pub mod kvaerno;

/// Butcher Tableau structure for Runge-Kutta methods.
/// 
/// A Butcher tableau encodes the coefficients of a Runge-Kutta method and provides the
/// necessary components for solving ordinary differential equations. This implementation
/// includes support for embedded methods (for error estimation) and dense output through interpolation.
/// 
/// # Generic Parameters
/// - `T`: The type of the coefficients, typically a floating-point type (e.g., `f32`, `f64`).
/// - `S`: Number of stages in the method.
/// - `I`: Primary Stages plus extra stages for interpolation (default is equal to `S`).
/// 
/// # Fields
/// - `c`: Node coefficients (time steps within the interval).
/// - `a`: Runge-Kutta matrix coefficients (coupling between stages).
/// - `b`: Weight coefficients for the primary method's final stage.
/// - `bh`: Weight coefficients for the embedded method (used for error estimation).
/// - `bi`: Weight coefficients for the interpolation method (used for dense output).
/// - `er`: Error estimation coefficients (optional, not all adaptive methods have these).
///   
///   These allow approximation at any point within the integration step.
/// 
pub struct ButcherTableau<T: Real, const S: usize, const I: usize = S> {
    pub c: [T; I],
    pub a: [[T; I]; I],
    pub b: [T; S],
    pub bh: Option<[T; S]>,
    pub bi: Option<[[T; I]; I]>,
    pub er: Option<[T; S]>,
}

impl<T: Real, const S: usize, const I: usize> ButcherTableau<T, S, I> {
    /// Number of stages in the method
    pub const STAGES: usize = S;

    /// Number of extra stages for interpolation
    pub const EXTRA_STAGES: usize = I - S;
}