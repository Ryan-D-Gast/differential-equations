//! Example 06: Numerical Integration
//!
//! This example shows how ODEs can be used for numerical integration by solving:
//! dy/dt = f(t)
//!
//! where:
//! - f(t) is the function to be integrated
//! - y(t) gives the integral of f(t) from t₀ to t
//!
//! In this example, we integrate f(t) = t, which has the analytical solution y(t) = t²/2 + C,
//! where C is the integration constant (set by the initial condition).
//!
//! This demonstrates:
//! - Using ODE solvers for numerical integration
//! - Comparing numerical results with analytical solutions
//! - Different output options: dense, even, and t-eval
//! - Error assessment in numerical integration

use differential_equations::ivp::IVP;
use differential_equations::prelude::*;

#[derive(Clone)]
struct IntegrationODE;

impl ODE for IntegrationODE {
    fn diff(&self, t: f64, _y: &f64, dydt: &mut f64) {
        *dydt = t;
    }
}

fn main() {
    // --- Problem Configuration ---
    let ode = IntegrationODE;
    let t0 = 0.0;
    let tf: f64 = 5.0;
    let y0 = 0.0;

    // --- Solve the ODE ---
    let solution = IVP::ode(&ode, t0, tf, y0)
        .method(ExplicitRungeKutta::rkf45())
        .solve()
        .unwrap();

    // Print the results.
    println!("Numerical Integration Example:");
    println!("-----------------------------");
    println!("t\t\ty");

    for (t, y) in solution.iter() {
        println!("{:.6}\t{:.6}", t, y);
    }

    // Verify the result. The analytical solution of y' = t with y(0) = 0 is y = t^2 / 2.
    let analytical_solution = tf.powi(2) / 2.0;
    let numerical_solution = solution.y.last().unwrap();
    let error = (analytical_solution - numerical_solution).abs();

    println!("-----------------------------");
    println!("Analytical Solution at tf: {:.6}", analytical_solution);
    println!("Numerical Solution at tf: {:.6}", numerical_solution);
    println!("Absolute Error: {:.6}", error);

    // Example with dense output
    println!("-----------------------------");
    println!("Dense Output Example:");
    let solution_dense = IVP::ode(&ode, t0, tf, y0)
        .dense(2) // 5 interpolation points between each step
        .method(ExplicitRungeKutta::rkf45())
        .solve()
        .unwrap();

    println!("t\t\ty");
    for (t, y) in solution_dense.iter() {
        println!("{:.6}\t{:.6}", t, y);
    }

    // Example with even t-out
    println!("-----------------------------");
    println!("Even t-out Example:");
    let solution_even = IVP::ode(&ode, t0, tf, y0)
        .even(1.0) // t-out at interval dt: 1.0
        .method(ExplicitRungeKutta::rkf45())
        .solve()
        .unwrap();

    println!("t\t\ty");
    for (t, y) in solution_even.iter() {
        println!("{:.6}\t{:.6}", t, y);
    }

    // Example with t-out points
    println!("-----------------------------");
    println!("t-out Points Example:");
    let t_out = vec![0.0, 2.0, 5.0];
    let solution_t_out = IVP::ode(&ode, t0, tf, y0)
        .t_eval(t_out)
        .method(ExplicitRungeKutta::rkf45())
        .solve()
        .unwrap();

    println!("t\t\ty");
    for (t, y) in solution_t_out.iter() {
        println!("{:.6}\t{:.6}", t, y);
    }
}
