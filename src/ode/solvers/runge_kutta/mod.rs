//! Runge-Kutta methods for solving ordinary differential equations.

// Contains macros to create Fixed step and Adaptive step Runge-Kutta methods.
mod macros;

mod fixed_step;
pub use fixed_step::{
    Euler,       // Euler's method (1st order Runge-Kutta)
    Midpoint,    // Midpoint method (2nd order Runge-Kutta)
    Heun,        // Heun's method (2nd order Runge-Kutta)
    Ralston,     // Ralston's method (2nd order Runge-Kutta)
    RK4,         // Classical 4th order Runge-Kutta method
    ThreeEights  // 3/8 Rule 4th order Runge-Kutta method
};

mod adaptive_step;
pub use adaptive_step::{
    RKF,        // Runge-Kutta-Fehlberg 4(5) adaptive method
    CashKarp,   // Cash-Karp 4(5) adaptive method
    DOPRI5      // Dormand-Prince 5(4) adaptive method
};

// Specialized Solvers for ODEs, These are NOT generated by Macros due to unique features and optimizations.
mod dop853;

pub use dop853::{
    DOP853, // Explicit Runge-Kutta method of order 8 with 5th order error estimate and 3rd order interpolant
};