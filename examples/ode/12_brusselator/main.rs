//! # Example 12: Implicit Solver with Brusselator System
//!
//! This example demonstrates solving the stiff Brusselator system
//! using an implicit Runge-Kutta method (Gauss-Legendre 6th order) with both an
//!
//! The Brusselator system is:
//! dy0/dt = 1 - 4*y0 + y0^2 * y1
//! dy1/dt = 3*y0 - y0^2 * y1
//!
//! Initial conditions: y0(0) = 1.5, y1(0) = 3.0
//!
//! The jacobian matrix J is:
//! J = [[-4 + 2*y0*y1,     y0^2],
//!      [ 3 - 2*y0*y1,    -y0^2]]

use differential_equations::prelude::*;
use nalgebra::Vector2;

struct BrusselatorSystem;

impl ODE<f64, Vector2<f64>> for BrusselatorSystem {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        let y0 = y[0];
        let y1 = y[1];

        dydt[0] = 1.0 - 4.0 * y0 + y0 * y0 * y1;
        dydt[1] = 3.0 * y0 - y0 * y0 * y1;
    }

    fn jacobian(&self, _t: f64, y: &Vector2<f64>, j: &mut Matrix<f64>) {
        let y0 = y[0];
        let y1 = y[1];

        j[(0, 0)] = -4.0 + 2.0 * y0 * y1;
        j[(0, 1)] = y0 * y0;
        j[(1, 0)] = 3.0 - 2.0 * y0 * y1;
        j[(1, 1)] = -y0 * y0;
    }
}

fn main() {
    // --- Problem Configuration ---
    let y0 = Vector2::new(1.5, 3.0);
    let t0 = 0.0;
    let tf = 20.0;
    let ode = BrusselatorSystem;
    let problem = ODEProblem::new(ode, t0, tf, y0);

    // --- Solve the ODE ---
    let mut method = ImplicitRungeKutta::gauss_legendre_6()
        .rtol(1e-6)
        .atol(1e-6)
        .h0(1e-3)
        .max_newton_iter(20); // Set maximum Newton iterations, only for implicit methods
    match problem.even(0.5).solve(&mut method) {
        Ok(solution) => {
            // Print the solution
            println!("Solution successfully obtained.");
            println!("Status: {:?}", solution.status);
            println!("Solution points (t, y0, y1):");
            for (t, y_val) in solution.iter() {
                println!("t: {:.4}, y0: {:.4}, y1: {:.4}", t, y_val[0], y_val[1]);
            }

            // Print statistics
            println!("\nStatistics:");
            println!("  Function evaluations: {}", solution.evals.function);
            println!("  jacobian evaluations: {}", solution.evals.jacobian);
            println!("  Newton iterations: {}", solution.evals.newton);
            println!("  Total steps taken: {}", solution.steps.total());
            println!("  Accepted steps: {}", solution.steps.accepted);
            println!("  Rejected steps: {}", solution.steps.rejected);
        }
        Err(e) => {
            eprintln!("An error occurred: {:?}", e);
        }
    }
}
