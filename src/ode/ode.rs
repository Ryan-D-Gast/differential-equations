//! Defines system of differential equations for numerical solvers.
//! The NumericalMethods use this trait to take a input system from the user and solve
//! Includes a differential equation and optional event function to interupt solver
//! given a condition or event.

use crate::{
    linalg::Matrix,
    traits::{DefaultState, Real, State},
};

/// ODE Trait for Differential Equations
///
/// ODE trait defines the differential equation dydt = f(t, y) for the solver.
/// The differential equation is used to solve the ordinary differential equation.
/// The trait also includes a solout function to interupt the solver when a condition
/// is met or event occurs.
///
/// # Impl
/// * `diff`    - Differential Equation dydt = f(t, y) in form f(t, &y, &mut dydt).
/// * `event`   - Event function to interupt solver when condition is met or event occurs.
/// * `jacobian` - Jacobian matrix J = df/dy for the system of equations.
///
/// Note that the event and jacobian functions are optional and can be left out when implementing.
pub trait ODE<T = f64, Y = DefaultState<T>>
where
    T: Real,
    Y: State<T>,
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
    fn diff(&self, t: T, y: &Y, dydt: &mut Y);

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
    fn jacobian(&self, t: T, y: &Y, j: &mut Matrix<T>) {
        if self.jacobian_ad(t, y, j) {
            return;
        }

        // Default implementation using forward finite differences
        let dim = y.len();
        let mut y_perturbed = y.clone();
        let mut f_perturbed = y.zeros_like();
        let mut f_origin = y.zeros_like();

        // Compute the unperturbed derivative
        self.diff(t, y, &mut f_origin);

        // Use sqrt of machine epsilon for finite differences
        let eps = T::default_epsilon().sqrt();

        // For each column of the jacobian
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

    /// Evaluates the exact Jacobian matrix using automatic differentiation (AD).
    ///
    /// This method is optional and returns `true` if AD was used, `false` otherwise.
    /// It provides exact partial derivatives utilizing the `num-dual` crate for better
    /// performance in stiff solvers by preventing round-off errors.
    ///
    /// The default implementation simply returns `false` to indicate that AD is not available
    /// and triggers a fallback to finite differences. Users can override it by using `num_dual::Dual64`
    /// or similar dual number types to evaluate the `ODE::diff` method.
    ///
    /// # Arguments
    /// * `t` - Independent variable grid point.
    /// * `y` - Dependent variable vector.
    /// * `j` - jacobian matrix.
    ///
    #[allow(unused_variables)]
    fn jacobian_ad(&self, t: T, y: &Y, j: &mut Matrix<T>) -> bool {
        false
    }
}

// AD tests
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    #[cfg(feature = "num-dual")]
    use num_dual::Dual64;

    struct TestODE;

    impl<T: Real> ODE<T, SVector<T, 2>> for TestODE {
        fn diff(&self, _t: T, y: &SVector<T, 2>, dydt: &mut SVector<T, 2>) {
            dydt[0] = -y[0] + y[0] * y[1];
            dydt[1] = y[0] - y[1] * y[1];
        }

        #[cfg(feature = "num-dual")]
        fn jacobian_ad(&self, _t: T, _y: &SVector<T, 2>, _j: &mut Matrix<T>) -> bool {
            // Check if T is f64, then we can do AD over Dual64.
            // Since we can't specialize cleanly here for T=f64 without specialization feature,
            // we typically let specific wrapper implementations handle AD. But for testing
            // AD manually here:
            false
        }
    }

    #[test]
    fn test_jacobian_fallback() {
        let ode = TestODE;
        let y = SVector::from([1.0, 2.0]);
        let mut j = Matrix::zeros(2, 2);
        ode.jacobian(0.0, &y, &mut j);

        // df0/dy0 = -1 + y[1] = 1.0
        // df0/dy1 = y[0] = 1.0
        // df1/dy0 = 1.0
        // df1/dy1 = -2*y[1] = -4.0

        assert!((j[(0, 0)] - 1.0).abs() < 1e-4);
        assert!((j[(0, 1)] - 1.0).abs() < 1e-4);
        assert!((j[(1, 0)] - 1.0).abs() < 1e-4);
        assert!((j[(1, 1)] - (-4.0)).abs() < 1e-4);
    }

    #[cfg(feature = "num-dual")]
    #[test]
    fn test_jacobian_ad_exact() {
        // Implement a specific AD wrapper to test the mechanism
        struct AdTestODE;

        impl ODE<f64, SVector<f64, 2>> for AdTestODE {
            fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
                dydt[0] = -y[0] + y[0] * y[1];
                dydt[1] = y[0] - y[1] * y[1];
            }

            fn jacobian_ad(&self, _t: f64, y: &SVector<f64, 2>, j: &mut Matrix<f64>) -> bool {
                let diff_dual = |dual_y: &SVector<Dual64, 2>| {
                    let mut dydt = SVector::<Dual64, 2>::zeros();
                    dydt[0] = -dual_y[0] + dual_y[0] * dual_y[1];
                    dydt[1] = dual_y[0] - dual_y[1] * dual_y[1];
                    dydt
                };

                for j_col in 0..2 {
                    let mut dual_y = y.map(|v| Dual64::new(v, 0.0));
                    dual_y[j_col].eps = 1.0;

                    let dual_dydt = diff_dual(&dual_y);

                    for i_row in 0..2 {
                        j[(i_row, j_col)] = dual_dydt[i_row].eps;
                    }
                }

                true
            }
        }

        let ode = AdTestODE;
        let y = SVector::from([1.0, 2.0]);
        let mut j = Matrix::zeros(2, 2);

        // This will call jacobian_ad, which returns true and populates j exactly
        ode.jacobian(0.0, &y, &mut j);

        assert_eq!(j[(0, 0)], 1.0);
        assert_eq!(j[(0, 1)], 1.0);
        assert_eq!(j[(1, 0)], 1.0);
        assert_eq!(j[(1, 1)], -4.0);
    }
}
