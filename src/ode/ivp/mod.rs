//! Initial Value Problem (IVP)

// Definitions & Constructors for users to ergonomically solve an IVP problem via the solve_ivp function.
mod ivp;
pub use ivp::IVP;

// Solve IVP function
mod solve_ivp;
//mod stepper;

pub use solve_ivp::solve_ivp;
//pub use stepper::Stepper;