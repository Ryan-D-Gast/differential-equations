// Implicit Runge-Kutta methods

mod fixed_step;
pub use fixed_step::{
    BackwardEuler, // Backward Euler method (1st order Runge-Kutta)
    CrankNicolson, // Crank-Nicolson method (2nd order Runge-Kutta)
};

mod adaptive_step;
pub use adaptive_step::{
    GaussLegendre4, // Gauss-Legendre 4th order method (implicit Runge-Kutta)
    GaussLegendre6, // Gauss-Legendre 6th order method (implicit Runge-Kutta)
};