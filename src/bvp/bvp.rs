use crate::traits::{DefaultState, Real, State};

/// Boundary Value Problem (BVP) Trait
///
/// Defines the differential equation and the boundary conditions for the solver.
pub trait BVP<T = f64, Y = DefaultState<T>>
where
    T: Real,
    Y: State<T>,
{
    /// Differential Equation dydt = f(t, y)
    fn diff(&self, t: T, y: &Y, dydt: &mut Y);

    /// Boundary conditions.
    /// The function should compute the residual: `res = g(y_a, y_b)`.
    /// `y_a` is the state at the initial point `t0` (or `a`), and `y_b` is the state at the final point `tf` (or `b`).
    /// The solver seeks to find a solution where `res == 0`.
    fn bound(&self, y_a: &Y, y_b: &Y, res: &mut Y);
}

impl<EqType, T: Real, Y: State<T>> BVP<T, Y> for &EqType
where
    EqType: BVP<T, Y>,
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        (*self).diff(t, y, dydt)
    }

    fn bound(&self, y_a: &Y, y_b: &Y, res: &mut Y) {
        (*self).bound(y_a, y_b, res)
    }
}
