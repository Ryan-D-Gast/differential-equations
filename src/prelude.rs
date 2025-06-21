//! Prelude module for the Differential Equations library.
//!
//! This prelude re-exports the most commonly used types and solvers
//! for Ordinary Differential Equations (ODEs), Delay Differential Equations (DDEs),
//! and Stochastic Differential Equations (SDEs).
//!
//! ## Solver Naming Conventions
//!
//! Some of the solvers, due to having the same name as another solver (e.g., `DOPRI5`
//! is used for both ODE and DDE), are given aliases for non-ODE versions.
//! For instance, the DDE version of `DOPRI5` is aliased as `DDE45`.
//! The convention for aliases is typically `[EquationType][Order(s)]`,
//! e.g., `DDE45` for a 4th/5th order DDE solver.
//!
//! As ODEs are by far the most common type of differential equation, the ODE
//! version of a solver is given priority in naming (i.e., it keeps the original name).
//! The original name of any aliased solver can be found by navigating to its module
//! within the library. This approach allows more solvers to be exposed
//! in the prelude without naming conflicts.
//!
//! Note that the naming convention for non-aliased solvers is often based on
//! their historical names. For example, `DOPRI5` is an adaptive step size
//! solver, and its name reflects its origin.
//!
//! ## Available Solvers
//!
//! ### Ordinary Differential Equations (ODE)
//!
//! #### Explicit Runge-Kutta Methods:
//! - `DOP853`: Adaptive Step Dormand-Prince 8(5,3) with dense output of order 7.
//! - `DOPRI5`: Adaptive Step Dormand-Prince 5(4).
//! - `Euler`: Fixed Step Euler method.
//! - `RK4`: Fixed Step Runge-Kutta 4th Order.
//! - `RKF`: Fixed Step Runge-Kutta-Fehlberg.
//! - `RKV65`: Verner 6(5) adaptive method with dense output of order 5.
//! - `RKV98`: Verner 9(8) adaptive method with dense output of order 9.
//!
//! #### Implicit Runge-Kutta Methods:
//! - `GaussLegendre6`: Gauss-Legendre 6th order method.
//!
//! ### Delay Differential Equations (DDE)
//! - `DDE23` (alias for `BS23`): Bogacki-Shampine 2(3) adaptive method with dense output.
//! - `DDE45` (alias for `DOPRI5`): Dormand-Prince 5(4) adaptive method with dense output.
//!
//! ### Stochastic Differential Equations (SDE)
//! - `EM`: Euler-Maruyama method (strong order 0.5).
//! - `Milstein`: Milstein method (strong order 1.0).
//! - `RKM4`: Stochastic Runge-Kutta Maruyama 4.
//!
//! For detailed examples, including problem setup and full solution process,
//! please refer to the `examples` directory in the repository.

// -- Types of Differential Equations --

pub use crate::{
    // Numerical Methods
    methods::{
        ExplicitRungeKutta, // Explicit Runge-Kutta methods for ODEs, DDEs, and SDEs
        ImplicitRungeKutta, // Implicit Runge-Kutta methods for ODEs
    },

    // Ordinary Differential Equations (ODE) module
    ode::{
        ODE,        // Define the ODE system
        ODEProblem, // Create IVP to solve

        // Re-exporting popular ODE solvers
        methods::runge_kutta::{
            explicit::{
                DOP853, // Adaptive Step Dormand-Prince 8(5,3) NumericalMethod with dense output of order 7
                DOPRI5, // Adaptive Step Dormand-Prince 5(4) NumericalMethod
                Euler,  // Fixed Step Euler NumericalMethod
                RK4,    // Fixed Step Runge-Kutta 4th Order NumericalMethod
                RKF,    // Fixed Step Runge-Kutta-Fehlberg NumericalMethod
                RKV65,  // Verner 6(5) adaptive method with dense output of order 5
                RKV87,  // Verner 8(7) adaptive method with dense output of order 7
                RKV98,  // Verner 9(8) adaptive method with dense output of order 9
            },
            implicit::{
                GaussLegendre6, // Gauss-Legendre 6th order method (implicit Runge-Kutta)
            },
        },
    },

    // Delay Differential Equations (DDE) module
    dde::{
        DDE,        // Define the DDE system
        DDEProblem, // Create IVP to solve

        // Re-exporting popular DDE solvers
        methods::{
            BS23 as DDE23, // Bogacki-Shampine 2(3) method with adaptive step size and dense output
            DOPRI5 as DDE45, // Dormand-Prince 5(4) method with adaptive step size
        },
    },

    // Stochastic Differential Equations (SDE) module
    sde::{
        SDE,        // Define the SDE system
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
        CrossingDirection,
        Solout, // Trait for defining a custom output behavior
    },
    alias::Evals,
    control::ControlFlag,
    derive::State,
    error::Error,
    interpolate::Interpolation,
    solution::Solution,
    status::Status,
};

// -- re-export of nalgebra Types and Macros --

pub use nalgebra::{
    DMatrix,
    SMatrix,
    SVector,
    vector,
};