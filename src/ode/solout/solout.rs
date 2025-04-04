//! Solout trait to choose which points to output during the solving process.

use super::*;

pub trait Solout<T, const R: usize, const C: usize, E = String>
where
    T: Real,
    E: EventData
{
    /// Solout function to choose which points to output during the solving process.
    /// 
    /// # Arguments
    /// * `solver` - Reference to the solver to use for solving the IVP.
    /// * `solution` - Immutable reference to the solution struct to avoid ownership issues.
    /// 
    fn solout<S>(&mut self, solver: &mut S, solution: &mut Solution<T, R, C, E>)
    where 
        S: Solver<T, R, C, E>;

    /// Tells solver if to include t0 and tf by appending them to the output vectors.
    /// 
    /// By default, this returns true as typically we want to include t0 and tf in the output.
    /// Thus the user can usually ignore implementing this function unless they want to exclude t0 and tf.
    /// 
    fn include_t0_tf(&self) -> bool {
        true
    }
}