//! Solvers for Ordinary Differential Equations (ODEs).

// Adams Methods for solving ordinary differential equations.
pub mod adams;

// Runge-Kutta methods for solving ordinary differential equations.
pub mod runge_kutta;

// Utilities for solvers to help standardize and reduce code duplication.
pub mod utils;

pub use {
    // Runge-Kutta Methods Generated by Macros, Due to this a few extra calculation could be done
    // during calculation reducing efficiency. On high levels of optimization this should be optimized and
    // a non factor. The advantage is that many methods can be generated with little code and a reduction in
    // the required number of tests as once the macro is tested the only source of error is the butcher tableau.
    runge_kutta::{
        // Fixed Step Size
        Euler,       // Euler's method (1st order Runge-Kutta)
        Midpoint,    // Midpoint method (2nd order Runge-Kutta)
        Heun,        // Heun's method (2nd order Runge-Kutta)
        Ralston,     // Ralston's method (2nd order Runge-Kutta)
        RK4,         // Classical 4th order Runge-Kutta method
        ThreeEights, // 3/8 Rule 4th order Runge-Kutta method
        // Adaptive Step Size
        RKF,         // Runge-Kutta-Fehlberg 4(5) adaptive method
        CashKarp,    // Cash-Karp 4(5) adaptive method
        DOPRI5,      // Dormand-Prince 5(4) adaptive method

        // Unique implementations of Runge-Kutta methods (non-macro)
        DOP853,      // Explicit Runge-Kutta method of order 8 with 5th order error estimate and 3rd order interpolant

        // Dense output methods for adaptive Runge-Kutta methods
        Verner65,    // Verner 6(5) adaptive method with dense output
        Verner98,    // Verner 9(8) adaptive method with dense output
    },

    // Adams Predictor Corrector Methods.
    adams::{
        // Fixed Step Size
        APCF4,   // Adams-Predictor-Corrector 4th Order Fixed Step Size Method
        // Adaptive Step Size
        APCV4,   // Adams-Predictor-Corrector 4th Order Variable Step Size Method
    },
};