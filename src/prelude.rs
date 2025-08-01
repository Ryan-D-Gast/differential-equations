//! Prelude module for the Differential Equations library.
//!
//! This prelude re-exports all the essential types and methods
//! for Ordinary Differential Equations (ODEs), Delay Differential Equations (DDEs),
//! and Stochastic Differential Equations (SDEs).
//!
//! ## Solver Naming Conventions
//!
//! Solvers are accessed through their respective method modules using the pattern
//! `MethodType::SolverName`. For example, `ExplicitRungeKutta::dop853` or
//! `ImplicitRungeKutta::gauss_legendre6`. This approach provides clear organization
//! and avoids naming conflicts between different equation types.
//!
//! ## Sample of Available Solvers
//!
//! ### Explicit Runge-Kutta Methods:
//! - `ExplicitRungeKutta::dop853`: Adaptive Step Dormand-Prince 8(5,3) with dense output of order 7.
//! - `ExplicitRungeKutta::dopri5`: Adaptive Step Dormand-Prince 5(4) with dense output of order 4.
//! - `ExplicitRungeKutta::euler`: Fixed Step Euler method.
//! - `ExplicitRungeKutta::rk4`: Fixed Step Runge-Kutta 4th Order.
//! - `ExplicitRungeKutta::rkv65e`: Verner's efficient 6(5) adaptive method with dense output of order 5.
//! - `ExplicitRungeKutta::rkv98e`: Verner's efficient 9(8) adaptive method with dense output of order 9.
//!
//! ODEs and DDEs are supported by these methods.
//! SDEs are supported for fixed step methods such as Euler and RK4.
//!
//! ### Implicit Runge-Kutta Methods:
//! - `ImplicitRungeKutta::gauss_legendre6`: Gauss-Legendre 6th order method.
//!
//! For detailed examples, including problem setup and full solution process,
//! please refer to the `examples` directory in the repository.

// -- Types of Differential Equations --

pub use crate::{
    // Numerical Methods
    methods::{
        ExplicitRungeKutta, // Explicit Runge-Kutta methods for ODEs, DDEs, and SDEs
        ImplicitRungeKutta, // Implicit Runge-Kutta methods for ODEs
        AdamsPredictorCorrector, // Adams Predictor-Corrector methods for ODEs
    },

    // Ordinary Differential Equations (ODE) module
    ode::{
        ODE,        // Define the ODE system
        ODEProblem, // Create IVP to solve
    },

    // Delay Differential Equations (DDE) module
    dde::{
        DDE,        // Define the DDE system
        DDEProblem, // Create IVP to solve
    },

    // Stochastic Differential Equations (SDE) module
    sde::{
        SDE,        // Define the SDE system
        SDEProblem, // Create IVP to solve
    },

    // Shared items not specific to a Differential Equation Type
    solout::{
        CrossingDirection,
        Solout, // Trait for defining a custom output behavior
    },
    stats::Evals,
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