//! Defines system of differential algebraic equations for numerical solvers.
//! The NumericalMethods use this trait to take a input system from the user and solve
//! Includes differential equations, mass matrix, and optional event function to interrupt solver
//! given a condition or event.

use crate::{
    linalg::Matrix,
    traits::{DefaultState, Real, State},
};

/// DAE Trait for Differential Algebraic Equations
///
/// DAE trait defines the differential algebraic equation system m·y' = f(t, y) for the solver.
/// Where m is the mass matrix, y is the solution vector, y' is the derivative vector, and
/// f(t, y) is the right-hand side function. When m is the identity matrix, this reduces
/// to the standard ODE form y' = f(t, y). The trait also includes an event function to
/// interrupt the solver when a condition is met or event occurs.
///
/// # Impl
/// * `diff`  - Right-hand side function f(t, y) in form f(t, &y, &mut f).
/// * `mass`  - Mass matrix m that multiplies y' in the DAE system m·y' = f(t, y).
/// * `event` - Event function to interrupt solver when condition is met or event occurs.
///
/// Note that the event function is optional and can be left out when implementing
/// in which case it will be set to return Continue by default.
pub trait DAE<T = f64, V = DefaultState<T>>
where
    T: Real,
    V: State<T>,
{
    /// Right-hand side function f(t, y)
    ///
    /// The right-hand side function f(t, y) represents the forcing terms in the
    /// differential algebraic equation system m·y' = f(t, y). This function
    /// computes the vector f given the current time t and solution vector y.
    ///
    /// For efficiency and ergonomics, the right-hand side is calculated using a
    /// mutable reference to the result vector. This allows for calculations to be
    /// performed in place, which is more efficient for iterative DAE solvers.
    ///
    /// # Arguments
    /// * `t`   - Independent variable point (usually time).
    /// * `y`   - Dependent variable vector (solution).
    /// * `f`   - Right-hand side vector f(t, y).
    ///
    fn diff(&self, t: T, y: &V, f: &mut V);

    /// Mass matrix m
    ///
    /// The mass matrix m appears in the differential algebraic equation system
    /// m·y' = f(t, y). The mass matrix characterizes which equations are differential
    /// (where the corresponding row has non-zero entries) and which are algebraic
    /// (where the corresponding row is zero or nearly singular).
    ///
    /// Special cases:
    /// - Identity matrix: Reduces to standard ODE y' = f(t, y)
    /// - Singular matrix: True DAE system with algebraic constraints
    /// - Constant matrix: Linear DAE system
    ///
    /// For efficiency, the mass matrix is calculated using a mutable reference.
    ///
    /// # Arguments
    /// * `m` - Mass matrix M. This matrix should be pre-sized by the caller to `dim x dim` where `dim = y.len()`.
    ///
    fn mass(&self, m: &mut Matrix<T>);

    /// Jacobian matrix j = ∂f/∂y
    ///
    /// The Jacobian matrix with respect to y is a matrix of partial derivatives
    /// of the right-hand side function F with respect to the solution variables Y.
    /// This matrix describes the sensitivity of the right-hand side to changes
    /// in the solution and is essential for implicit DAE solvers.
    ///
    /// By default, this method uses a finite difference approximation.
    /// Users can override this with an analytical implementation for better efficiency.
    ///
    /// # Arguments
    /// * `t` - Independent variable grid point.
    /// * `y` - Dependent variable vector.
    /// * `j` - Jacobian matrix ∂F/∂Y. This matrix should be pre-sized by the caller to `dim x dim` where `dim = y.len()`.
    ///
    fn jacobian(&self, t: T, y: &V, j: &mut Matrix<T>) {
        // Default implementation using forward finite differences
        let dim = y.len();
        let mut y_perturbed = y.clone();
        let mut f_perturbed = y.zeros_like();
        let mut f_origin = y.zeros_like();

        // Compute the unperturbed right-hand side
        self.diff(t, y, &mut f_origin);

        // Use sqrt of machine epsilon for finite differences
        let eps = T::default_epsilon().sqrt();

        // For each column of the Jacobian
        for j_col in 0..dim {
            // Get the original value
            let y_original_j = y.get_component(j_col);

            // Calculate perturbation size (max of component magnitude or 1.0)
            let perturbation = eps * y_original_j.abs().max(T::one());

            // Perturb the component
            y_perturbed.copy_from_state(y);
            y_perturbed.set_component(j_col, y_original_j + perturbation);

            // Evaluate function with perturbed value
            self.diff(t, &y_perturbed, &mut f_perturbed);

            // Compute finite difference approximation for this column
            for i_row in 0..dim {
                j[(i_row, j_col)] = (f_perturbed.get_component(i_row)
                    - f_origin.get_component(i_row))
                    / perturbation;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_dae_from_fn() {
        let t0 = 0.0;
        let tf = 1.0;
        let y0 = [1.0, 1.0];

        let solution = IVP::dae_from_fn(
            |_t, y, f| {
                f[0] = -y[0];
                f[1] = y[0] - y[1];
            },
            |m| {
                m[(0, 0)] = 1.0;
                m[(1, 1)] = 0.0; // y[1] is algebraic
            },
            t0, tf, y0
        )
        .method(ImplicitRungeKutta::radau5())
        .solve()
        .unwrap();

        let final_y = solution.y.last().unwrap();
        assert!((final_y[0] - 0.3678).abs() < 1e-1);
    }
}
