//! Prelude module for the Differential Equations library.

// -- Types of Differential Equations --

pub use crate::{
    // Ordinary Differential Equations (ODE) module
    ode::{
        ODE,        // Define the ODE system
        ODEProblem, // Create IVP to solve

        // Re-exporting popular ODE solvers
        methods::{
            DOP853, // Adaptive Step Dormand-Prince 8(5,3) NumericalMethod with dense output of order 7
            DOPRI5, // Adaptive Step Dormand-Prince 5(4) NumericalMethod
            RK4,    // Fixed Step Runge-Kutta 4th Order NumericalMethod
            RKV65,  // Verner 6(5) adaptive method with dense output of order 5
            RKV98,  // Verner 9(8) adaptive method with dense output of order 9  RKV98,
        },
    },

    // Stochastic Differential Equations (SDE) module
    sde::{
        SDE, // Define the SDE system
        SDEProblem, // Create IVP to solve

        // Re-exporting popular SDE solvers
        methods::{
            EM,       // Euler-Maruyama method (strong order 0.5)
            Milstein, // Milstein method (strong order 1.0)
            RKM4,     // Stochastic Runge-Kutta Maruyama 4
        },
    },

    // Shared items not specific to a Differential Equation Type
    solout::{
        Solout, // Trait for defining a custom output behavior
        CrossingDirection,
    },
    error::Error,
    status::Status,
    solution::Solution,
    control::ControlFlag,
    interpolate::Interpolation, 
    derive::State,
};