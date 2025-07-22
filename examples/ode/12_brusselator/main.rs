//! # Example 12: Implicit Solver with Brusselator System
//!
//! This example demonstrates solving the stiff Brusselator system
//! using an implicit Runge-Kutta method (Gauss-Legendre 6th order) with both an
//! analytically provided Jacobian (by calling `.jacobian()`) and a 
//! finite-difference approximated Jacobian (default).
//!
//! The Brusselator system is:
//! dy0/dt = 1 - 4*y0 + y0^2 * y1
//! dy1/dt = 3*y0 - y0^2 * y1
//!
//! Initial conditions: y0(0) = 1.5, y1(0) = 3.0
//!
//! The Jacobian matrix J is:
//! J = [[-4 + 2*y0*y1,     y0^2],
//!      [ 3 - 2*y0*y1,    -y0^2]]

use differential_equations::prelude::*;
use nalgebra::{DMatrix, Vector2};

// Define the ODE system: Brusselator
struct BrusselatorSystem;

impl ODE<f64, Vector2<f64>> for BrusselatorSystem {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        let y0 = y[0];
        let y1 = y[1];

        dydt[0] = 1.0 - 4.0 * y0 + y0 * y0 * y1;
        dydt[1] = 3.0 * y0 - y0 * y0 * y1;
    }

    fn jacobian(&self, _t: f64, y: &Vector2<f64>, j: &mut DMatrix<f64>) {
        let y0 = y[0];
        let y1 = y[1];

        j[(0, 0)] = -4.0 + 2.0 * y0 * y1;
        j[(0, 1)] = y0 * y0;
        j[(1, 0)] = 3.0 - 2.0 * y0 * y1;
        j[(1, 1)] = -y0 * y0;
    }
}

fn main() {
    // Analytical Jacobian run
    let mut method_analytical = ImplicitRungeKutta::gauss_legendre_6()
        .rtol(1e-6)
        .atol(1e-6)
        .h0(1e-3) // Initial step size suggestion
        .max_newton_iter(20); // Max Newton iterations

    // Initial conditions and time span for Brusselator
    let y0 = Vector2::new(1.5, 3.0); 
    let t0 = 0.0;
    let tf = 20.0; // Time span to observe oscillations

    // Define the ODE system
    let ode_analytical = BrusselatorSystem;

    // Create the ODE problem
    let problem_analytical = ODEProblem::new(ode_analytical, t0, tf, y0);

    // Solve the problem
    match problem_analytical
        .even(0.5) // Output points every 0.5 time units
        .solve(&mut method_analytical)
    {
        Ok(solution) => {
            println!("Solution successfully obtained (Analytical Jacobian).");
            println!("Status: {:?}", solution.status);

            // Print a few points from the solution
            println!("Solution points (t, y0, y1) - Analytical Jacobian:");
            for (t, y_val) in solution.iter() {
                println!("t: {:.4}, y0: {:.4}, y1: {:.4}", t, y_val[0], y_val[1]);
            }
            
            // Print statistics
            println!("\nStatistics (Analytical Jacobian):");
            println!("  Function evaluations: {}", solution.evals);
            println!("  Jacobian evaluations: {}", solution.jac_evals); // Using jac_evals field
            println!("  Total steps taken: {}", solution.steps);
            println!("  Accepted steps: {}", solution.accepted_steps);
            println!("  Rejected steps: {}", solution.rejected_steps);
        }
        Err(e) => {
            eprintln!("An error occurred during analytical Jacobian solution: {:?}", e);
        }
    }
}
