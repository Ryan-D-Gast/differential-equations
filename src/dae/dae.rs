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
        let mut y_values = vec![T::zero(); dim];
        let mut y_perturbed_values = vec![T::zero(); dim];
        let mut f_origin_values = vec![T::zero(); dim];
        let mut f_perturbed_values = vec![T::zero(); dim];
        y.copy_to_flat_slice(&mut y_values);
        f_origin.copy_to_flat_slice(&mut f_origin_values);

        // Use sqrt of machine epsilon for finite differences
        let eps = T::default_epsilon().sqrt();

        // For each column of the Jacobian
        for j_col in 0..dim {
            // Get the original value
            let y_original_j = y_values[j_col];

            // Calculate perturbation size (max of component magnitude or 1.0)
            let perturbation = eps * y_original_j.abs().max(T::one());

            // Perturb the component
            y_perturbed.copy_to_flat_slice(&mut y_perturbed_values);
            y_perturbed_values[j_col] = y_original_j + perturbation;
            y_perturbed.copy_from_flat_slice(&y_perturbed_values);

            // Evaluate function with perturbed value
            self.diff(t, &y_perturbed, &mut f_perturbed);
            f_perturbed.copy_to_flat_slice(&mut f_perturbed_values);

            // Restore original value
            y_perturbed_values[j_col] = y_original_j;
            y_perturbed.copy_from_flat_slice(&y_perturbed_values);

            // Compute finite difference approximation for this column
            for i_row in 0..dim {
                j[(i_row, j_col)] =
                    (f_perturbed_values[i_row] - f_origin_values[i_row]) / perturbation;
            }
        }
    }
}
