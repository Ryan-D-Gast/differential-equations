//! Initial Value Problem (ODEProblem)

// Definitions & Constructors for users to ergonomically solve an ODEProblem problem via the solve_ode function.
mod ode_problem;
pub use ode_problem::ODEProblem;

// Solve ODEProblem function
mod solve_ode;
pub use solve_ode::solve_ode;
