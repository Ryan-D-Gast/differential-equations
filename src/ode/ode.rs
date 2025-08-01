//! Defines system of differential equations for numerical solvers.
//! The NumericalMethods use this trait to take a input system from the user and solve
//! Includes a differential equation and optional event function to interupt solver
//! given a condition or event.

use crate::{
    ControlFlag,
    traits::{CallBackData, Real, State},
};
use nalgebra::DMatrix;

/// ODE Trait for Differential Equations
///
/// ODE trait defines the differential equation dydt = f(t, y) for the solver.
/// The differential equation is used to solve the ordinary differential equation.
/// The trait also includes a solout function to interupt the solver when a condition
/// is met or event occurs.
///
/// # Impl
/// * `diff`    - Differential Equation dydt = f(t, y) in form f(t, &y, &mut dydt).
/// * `event`   - Solout function to interupt solver when condition is met or event occurs.
///
/// Note that the solout function is optional and can be left out when implementing
/// in which can it will be set to return false by default.
///
#[allow(unused_variables)]
pub trait ODE<T = f64, V = f64, D = String>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    /// Differential Equation dydt = f(t, y)
    ///
    /// An ordinary differential equation (ODE) takes a independent variable
    /// which in this case is 't' as it is typically time and a dependent variable
    /// which is a vector of values 'y'. The ODE returns the derivative of the
    /// dependent variable 'y' with respect to the independent variable 't' as
    /// dydt = f(t, y).
    ///
    /// For efficiency and ergonomics the derivative is calculated from an argument
    /// of a mutable reference to the derivative vector dydt. This allows for a
    /// derivatives to be calculated in place which is more efficient as iterative
    /// ODE solvers require the derivative to be calculated at each step without
    /// regard to the previous value.
    ///
    /// # Arguments
    /// * `t`    - Independent variable point.
    /// * `y`    - Dependent variable point.
    /// * `dydt` - Derivative point.
    ///
    fn diff(&self, t: T, y: &V, dydt: &mut V);

    /// Event function to detect significant conditions during integration.
    ///
    /// Called after each step to detect events like threshold crossings,
    /// singularities, or other mathematically/physically significant conditions.
    /// Can be used to terminate integration when conditions occur.
    ///
    /// # Arguments
    /// * `t`    - Current independent variable point.
    /// * `y`    - Current dependent variable point.
    ///
    /// # Returns
    /// * `ControlFlag` - Command to continue or stop solver.
    ///
    fn event(&self, t: T, y: &V) -> ControlFlag<T, V, D> {
        ControlFlag::Continue
    }
    
    /// jacobian matrix J = df/dy
    /// 
    /// The jacobian matrix is a matrix of partial derivatives of a vector-valued function.
    /// It describes the local behavior of the system of equations and can be used to improve
    /// the efficiency of certain solvers by providing information about the local behavior
    /// of the system of equations.
    /// 
    /// By default, this method uses a finite difference approximation.
    /// Users can override this with an analytical implementation for better efficiency.
    /// 
    /// # Arguments
    /// * `t` - Independent variable grid point.
    /// * `y` - Dependent variable vector.
    /// * `j` - jacobian matrix. This matrix should be pre-sized by the caller to `dim x dim` where `dim = y.len()`.
    /// 
    fn jacobian(&self, t: T, y: &V, j: &mut DMatrix<T>) {
        // Default implementation using forward finite differences
        let dim = y.len();
        let mut y_perturbed = y.clone();
        let mut f_perturbed = V::zeros();
        let mut f_origin = V::zeros();
        
        // Compute the unperturbed derivative
        self.diff(t, y, &mut f_origin);
        
        // Use sqrt of machine epsilon for finite differences
        let eps = T::default_epsilon().sqrt();
        
        // For each column of the jacobian
        for j_col in 0..dim {
            // Get the original value
            let y_original_j = y.get(j_col);
            
            // Calculate perturbation size (max of component magnitude or 1.0)
            let perturbation = eps * y_original_j.abs().max(T::one());
            
            // Perturb the component
            y_perturbed.set(j_col, y_original_j + perturbation);
            
            // Evaluate function with perturbed value
            self.diff(t, &y_perturbed, &mut f_perturbed);
            
            // Restore original value
            y_perturbed.set(j_col, y_original_j);
            
            // Compute finite difference approximation for this column
            for i_row in 0..dim {
                j[(i_row, j_col)] = (f_perturbed.get(i_row) - f_origin.get(i_row)) / perturbation;
            }
        }
    }
}