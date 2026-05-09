use rayon::prelude::*;

use crate::error::Error;
use crate::solution::Solution;
use crate::traits::{Real, State};

/// Extension trait to solve a collection of IVPs in parallel.
pub trait ParallelSolve<T, Y>
where
    T: Real,
    Y: State<T>,
{
    /// Solves a collection of initial value problems in parallel using `rayon`.
    fn par_solve(self) -> Vec<Result<Solution<T, Y>, Error<T, Y>>>;
}

impl<I, T, Y> ParallelSolve<T, Y> for I
where
    T: Real + Send,
    Y: State<T> + Send,
    I: IntoParallelIterator,
    I::Item: Solver<T, Y> + Send,
{
    fn par_solve(self) -> Vec<Result<Solution<T, Y>, Error<T, Y>>> {
        self.into_par_iter().map(|ivp| ivp.solve_ivp()).collect()
    }
}

/// A trait representing something that can be solved.
/// This allows us to abstract over the specific IVP types (ODE, DAE, etc.)
pub trait Solver<T: Real, Y: State<T>> {
    fn solve_ivp(self) -> Result<Solution<T, Y>, Error<T, Y>>;
}
