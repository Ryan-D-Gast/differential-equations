//! Prelude module for the Differential Equations library.

// -- Types of Differential Equations --

pub use crate::{
    // Ordinary Differential Equations (ODE) module
    ode::{
        ODE, // Define the ODE system
        ODEProblem, // Create IVP to solve

        // Commonly used methods
        DOP853,
        DOPRI5,
        RK4,
        RKV65,
        RKV98,
    },

    // Stochastic Differential Equations (SDE) module
    sde::{
        SDE, // Define the SDE system
        SDEProblem, // Create IVP to solve

        // Commonly used methods
        EM,
        Milstein,
        RKM4,
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