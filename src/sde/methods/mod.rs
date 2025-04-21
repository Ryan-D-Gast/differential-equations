//! Numerical methods for solving stochastic differential equations (SDEs).

mod euler_maruyama;
mod runge_kutta_maruyama;
mod milstein;

pub use euler_maruyama::EM;
pub use milstein::Milstein;
pub use runge_kutta_maruyama::RKM4;