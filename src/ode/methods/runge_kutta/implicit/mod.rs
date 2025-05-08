// Implicit Runge-Kutta methods without jacobian using fixed point iteration instead.

mod fixed_step;
pub use fixed_step::{
    BackwardEuler, // Backward Euler method 1st order implicit Runge-Kutta
    CrankNicolson, // Crank-Nicolson method 2nd order implicit Runge-Kutta
};

mod adaptive_step;
pub use adaptive_step::{
    GaussLegendre4, // Gauss-Legendre 4th order method implicit Runge-Kutta
    GaussLegendre6, // Gauss-Legendre 6th order method implicit Runge-Kutta
};