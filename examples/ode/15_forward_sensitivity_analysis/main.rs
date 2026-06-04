//! Example 15: Forward Sensitivity Analysis (FSA)
//!
//! This example demonstrates how to compute sensitivities of an ODE solution
//! with respect to its parameters using the new structured API for
//! forward sensitivities.
//!
//! We use a simple decay model: dy/dt = -k * y
//! The parameter is `k`. We want to know how the final state `y(t_f)` changes
//! with respect to `k` (i.e., dy/dk).

use differential_equations::prelude::*;
use nalgebra::SVector;

/// A simple decay model: dy/dt = -k * y
struct Decay {
    k: f64,
}

impl ODE<f64, SVector<f64, 1>> for Decay {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = -self.k * y[0];
    }
}

impl ParametrizedODE<f64, SVector<f64, 1>, SVector<f64, 1>> for Decay {
    fn parameters(&self) -> SVector<f64, 1> {
        SVector::from([self.k])
    }

    fn jacobian_p(&self, _t: f64, y: &SVector<f64, 1>, j: &mut Matrix<f64>) {
        // df/dk = -y
        j[(0, 0)] = -y[0];
    }
}

fn main() {
    let decay = Decay { k: 1.0 };

    // Initial conditions: y(0) = 1.0.
    // The initial sensitivity S(0) = dy/dk(0) = 0.0 because the initial state does not depend on `k`.
    // The augmented state is [y, S].
    let y0_aug = SVector::<f64, 2>::from([1.0, 0.0]);

    // Time span
    let t0 = 0.0;
    let tf = 2.0;

    let method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8);

    // Solve the augmented system using Forward Sensitivity Analysis.
    let fsa_ode = ForwardSensitivityOde::new(&decay, SVector::from([0.0]));

    let solution = IVP::ode(&fsa_ode, t0, tf, y0_aug)
        .method(method)
        .solve()
        .unwrap();

    // Extract the final state
    let y_final_aug = solution.y.last().unwrap();
    let y_final = y_final_aug[0];
    let sens_final = y_final_aug[1];

    // Analytical solution for verification: y(t) = y0 * e^(-k*t)
    // dy/dk = -t * y0 * e^(-k*t)
    let k_f64 = 1.0;
    let tf_f64 = 2.0_f64;
    let expected_val = (-k_f64 * tf_f64).exp();
    let expected_sens = -tf_f64 * expected_val;

    println!("Forward Sensitivity Analysis (FSA)");
    println!("-----------------------------------------------------");
    println!("Numerical final state:       {:.8}", y_final);
    println!("Analytical final state:      {:.8}", expected_val);
    println!("Numerical sensitivity dy/dk: {:.8}", sens_final);
    println!("Analytical sensitivity dy/dk:{:.8}", expected_sens);

    let error_val = (y_final - expected_val).abs();
    let error_sens = (sens_final - expected_sens).abs();
    println!("State error:                 {:.2e}", error_val);
    println!("Sensitivity error:           {:.2e}", error_sens);
}
