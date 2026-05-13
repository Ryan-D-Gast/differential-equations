//! Boundary value problem methods.

pub mod shooting;

use crate::{
    bvp::Boundary,
    error::Error,
    ode::ODE,
    solout::Solout,
    solution::Solution,
    traits::{Real, State},
};

pub use shooting::{Shooting, SingleShooting};

/// Trait for boundary value problem solvers.
pub trait BVPMethod<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// Solve an ODE boundary value problem from an initial-state guess.
    fn solve<EqType, SoloutType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y,
        solout: &mut SoloutType,
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
        SoloutType: Solout<T, Y>;
}
