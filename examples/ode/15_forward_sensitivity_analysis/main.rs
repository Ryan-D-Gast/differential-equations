//! Example 15: Forward Sensitivity Analysis (FSA) using Dual Numbers
//!
//! This example demonstrates how to compute sensitivities of an ODE solution
//! with respect to its parameters using the `num-dual` crate.
//!
//! Forward sensitivity analysis calculates the gradient of the solver states y
//! with respect to system parameters p. By making our ODE implementation generic
//! over the `Real` trait, we can solve the system using dual numbers, which compute
//! both the state and its derivatives simultaneously without requiring an
//! explicitly augmented sensitivity matrix system.
//!
//! We use a simple decay model: dy/dt = -k * y
//! The parameter is `k`. We want to know how the final state `y(t_f)` changes
//! with respect to `k` (i.e., dy/dk).

use differential_equations::ivp::IVP;
use differential_equations::prelude::*;
use differential_equations::traits::Real;
use nalgebra::SVector;
use num_dual::Dual64;

/// A simple decay model: dy/dt = -k * y
/// We make the struct generic over `T` to support both `f64` and `Dual64`.
struct Decay<T> {
    k: T,
}

impl<T: Real> ODE<T, SVector<T, 1>> for Decay<T> {
    fn diff(&self, _t: T, y: &SVector<T, 1>, dydt: &mut SVector<T, 1>) {
        dydt[0] = -self.k * y[0];
    }
}

fn main() {
    // We want to evaluate the sensitivity with respect to `k`. The dual part of
    // `k` is 1.0 because dk/dk = 1.
    let k = Dual64::new(1.0, 1.0);
    let decay = Decay { k };

    // Initial conditions: y(0) = 1.0.
    // The dual part is 0.0 because the initial state does not depend on `k`.
    let y0 = SVector::from([Dual64::from(1.0)]);

    // Time span
    let t0 = Dual64::from(0.0);
    let tf = Dual64::from(2.0);

    // Specify the DOP853 scalar and state types so the solver uses Dual64.
    let method = ExplicitRungeKutta::<_, _, Dual64, SVector<Dual64, 1>, 8, 12, 16>::dop853()
        .rtol(Dual64::from(1e-8))
        .atol(Dual64::from(1e-8));

    // Solve the ODE. The solver handles Dual64 through the same Real trait used
    // by f64.
    let solution = IVP::ode(&decay, t0, tf, y0).method(method).solve().unwrap();

    // Extract the final state
    let y_final_dual = solution.y.last().unwrap()[0];

    // The real part is the actual solution value: y(tf)
    let y_final = y_final_dual.re;
    // The dual part is the derivative with respect to k: dy/dk(tf)
    let sens_final = y_final_dual.eps;

    // Analytical solution for verification: y(t) = y0 * e^(-k*t)
    // dy/dk = -t * y0 * e^(-k*t)
    let k_f64 = 1.0;
    let tf_f64 = 2.0_f64;
    let expected_val = (-k_f64 * tf_f64).exp();
    let expected_sens = -tf_f64 * expected_val;

    println!("Forward Sensitivity Analysis (FSA) using Dual Numbers");
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
