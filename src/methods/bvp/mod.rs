//! Boundary Value Problem (BVP) methods.

mod shooting;

pub use shooting::ShootingMethod;

use crate::{
    bvp::bvp::BVP,
    error::Error,
    solution::Solution,
    traits::{Real, State},
};

/// Trait for BVP solvers.
pub trait BVPMethod<T, Y>
where
    T: Real,
    Y: State<T>,
{
    fn solve<EqType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y,
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: BVP<T, Y> + ?Sized;
}
