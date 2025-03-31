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
    /// * `system` - Reference to the system of equations being solved.
    /// * `t_out` - Vector to store the output time points.
    /// * `y_out` - Vector to store the output state points.
    /// 
    fn solout<S, F>(&mut self, solver: &mut S, ode: &F, t_out: &mut Vec<T>, y_out: &mut Vec<SMatrix<T, R, C>>)
    where 
        F: ODE<T, R, C, E>,
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