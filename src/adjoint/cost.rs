use crate::traits::{Real, State};

/// Cost Function Trait for Adjoint Sensitivity Analysis
///
/// Defines the cost function for which the gradient needs to be calculated.
/// The cost function is defined as a sum of a continuous integral part and
/// a discrete set of evaluations at specific time points.
///
/// `g(t, y, p) = ∫ integrand(t, y, p) dt + Σ discrete(t_i, y_i, p)`
pub trait CostFunction<T: Real, Y: State<T>, P: State<T>> {
    /// Continuous cost function part evaluated at time `t`.
    /// Represents the integral of `g(t, y, p)` over time.
    fn integrand(&self, _t: T, _y: &Y, _p: &P) -> T {
        T::zero()
    }

    /// Discrete cost function part evaluated at specific time points `t_i`.
    /// Represents the sum of `h(t_i, y_i, p)` at discrete times.
    fn discrete(&self, _t: T, _y: &Y, _p: &P) -> T {
        T::zero()
    }
}
