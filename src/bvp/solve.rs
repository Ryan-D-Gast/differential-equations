use crate::{
    bvp::bvp::BVP,
    error::Error,
    solution::Solution,
    traits::{Real, State},
};

pub trait BVPMethod<T: Real, Y: State<T>> {
    fn solve<EqType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y, // Initial guess for y(t0)
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: BVP<T, Y>;
}
