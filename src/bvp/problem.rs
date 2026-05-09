use crate::{
    bvp::bvp::BVP,
    error::Error,
    solution::Solution,
    traits::{Real, State},
};

/// BvpProblem Builder.
///
/// This builder configures a BVP problem with given bounds and initial guess.
#[derive(Clone, Debug)]
pub struct BvpProblem<EqType, T: Real, Y: State<T>, Method> {
    pub equation: EqType,
    pub t0: T,
    pub tf: T,
    pub y_guess: Y,
    pub method: Method,
}

impl<'a, F, T: Real, Y: State<T>> BvpProblem<&'a F, T, Y, ()>
where
    F: BVP<T, Y>,
{
    pub fn new(system: &'a F, t0: T, tf: T, y_guess: Y) -> Self {
        Self {
            equation: system,
            t0,
            tf,
            y_guess,
            method: (),
        }
    }
}

impl<EqType, T: Real, Y: State<T>, Method> BvpProblem<EqType, T, Y, Method> {
    pub fn method<NewMethod>(self, method: NewMethod) -> BvpProblem<EqType, T, Y, NewMethod> {
        BvpProblem {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method,
        }
    }
}

impl<EqType, T: Real, Y: State<T>, Method> BvpProblem<EqType, T, Y, Method>
where
    EqType: BVP<T, Y>,
    Method: crate::bvp::solve::BVPMethod<T, Y>,
{
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        self.method
            .solve(&self.equation, self.t0, self.tf, &self.y_guess)
    }
}
