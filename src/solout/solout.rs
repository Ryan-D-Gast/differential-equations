//! Solout trait to choose which points to output during the solving process.

use super::*;

pub trait Solout<T, V, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Solout function to choose which points to output during the solving process.
    ///
    /// # Arguments
    /// * `t_curr` - Current time.
    /// * `t_prev` - Previous time.
    /// * `y_curr` - Current solution vector.
    /// * `y_prev` - Previous solution vector.
    /// * `interpolator` - Interpolator with an .interpolate(t) method to get the solution at a given time between t_prev and t_curr.
    /// * `solution` - Immutable reference to the solution struct to avoid ownership issues.
    ///
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &V,
        y_prev: &V,
        interpolator: &mut I,
        solution: &mut Solution<T, V, D>,
    ) -> ControlFlag<D>
    where
        I: Interpolation<T, V>;
}
