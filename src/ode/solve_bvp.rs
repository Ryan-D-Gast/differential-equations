//! Low-level ODE boundary value problem solve function.

use crate::{
    bvp::Boundary,
    error::Error,
    methods::bvp::BVPMethod,
    ode::ODE,
    solout::Solout,
    solution::Solution,
    traits::{Real, State},
};

/// Solve an ODE boundary value problem with the selected method.
pub fn solve_bvp<T, Y, Method, EqType, SoloutType>(
    method: &mut Method,
    problem: &EqType,
    t0: T,
    tf: T,
    y_guess: &Y,
    solout: &mut SoloutType,
) -> Result<Solution<T, Y>, Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    Method: BVPMethod<T, Y>,
    EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
    SoloutType: Solout<T, Y>,
{
    method.solve(problem, t0, tf, y_guess, solout)
}
