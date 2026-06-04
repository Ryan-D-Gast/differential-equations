//! Shooting methods for boundary value problems.

mod multiple;
mod single;

pub use multiple::MultipleShooting;
pub use single::SingleShooting;

/// Constructor namespace for shooting BVP methods.
#[derive(Clone, Copy, Debug)]
pub struct Shooting;

impl Shooting {
    /// Create a single-shooting BVP method from an ODE IVP solver.
    pub fn single<M>(ode_solver: M) -> SingleShooting<M> {
        SingleShooting::new(ode_solver)
    }

    /// Create a multiple-shooting BVP method from an ODE IVP solver.
    pub fn multiple<M>(ode_solver: M) -> MultipleShooting<M> {
        MultipleShooting::new(ode_solver)
    }
}
