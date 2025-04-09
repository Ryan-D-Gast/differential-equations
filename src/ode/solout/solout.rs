//! Solout trait to choose which points to output during the solving process.

use super::*;

pub trait Solout<T, const R: usize, const C: usize, D = String>
where
    T: Real,
    D: CallBackData,
{
    /// Solout function to choose which points to output during the solving process.
    ///
    /// # Arguments
    /// * `solver` - Reference to the solver to use for solving the IVP.
    /// * `solution` - Immutable reference to the solution struct to avoid ownership issues.
    ///
    fn solout<S>(&mut self, solver: &mut S, solution: &mut Solution<T, R, C, D>) -> ControlFlag<D>
    where
        S: Solver<T, R, C, D>;
}
