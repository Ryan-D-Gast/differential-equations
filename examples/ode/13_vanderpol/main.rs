//! # Example 13: Van der Pol oscillator
//!
//! This example demonstrates solving the stiff Van der Pol oscillator
//! using an implicit Runge-Kutta method (Radau 5th order) with an
//! adaptive step size.
//!
//! The Van der Pol system (written as a first-order system) is:
//! dy0/dt = y1
//! dy1/dt = ((1 - y0^2) * y1 - y0) / mu
//!
//! Initial conditions: y0(0) = 2.0, y1(0) = -0.66
//!
//! The jacobian matrix J is:
//! J = [[0, 1],
//!      [(-2*y0*y1 - 1)/mu,  (1 - y0^2)/mu]]

use differential_equations::prelude::*;
use nalgebra::Vector2;

/// Van der Pol ODE: y' = f(t, y)
struct VanderPol {
    mu: f64,
}

impl VanderPol {
    fn new(mu: f64) -> Self {
        Self { mu }
    }
}

impl ODE<f64, Vector2<f64>> for VanderPol {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        dydt[0] = y[1];
        dydt[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / self.mu;
    }

    fn jacobian(&self, _t: f64, y: &Vector2<f64>, j: &mut Matrix<f64>) {
        j[(0, 0)] = 0.0;
        j[(0, 1)] = 1.0;
        j[(1, 0)] = (-2.0 * y[0] * y[1] - 1.0) / self.mu;
        j[(1, 1)] = (1.0 - y[0] * y[0]) / self.mu;
    }
}

fn main() {
    // --- Problem Configuration ---
    let y0 = Vector2::new(2.0, -0.66);
    let t0 = 0.0;
    let tf = 2.0;
    let mu = 1.0e-6; // small parameter -> stiff
    let model = VanderPol::new(mu);
    let problem = ODEProblem::new(model, t0, tf, y0);

    // --- Solve the ODE ---
    let mut method = ImplicitRungeKutta::radau5()
        .rtol(1.0e-4)
        .atol(1.0e-4)
        .h0(1.0e-6);

    match problem.even(0.2).solve(&mut method) {
        Ok(solution) => {
            println!("Solution successfully obtained.");
            println!("Status: {:?}", solution.status);
            println!("Solution points (t, y0, y1):");
            for (t, y) in solution.iter() {
                println!("t: {:.4}, y0: {:.6}, y1: {:.6}", t, y[0], y[1]);
            }

            // Print statistics
            println!("\nStatistics:");
            println!("  Function evaluations: {}", solution.evals.function);
            println!("  Jacobian evaluations: {}", solution.evals.jacobian);
            println!("  Newton iterations: {}", solution.evals.newton);
            println!("  Total LU decompositions: {}", solution.evals.decompositions);
            println!("  Total Ax=b solves: {}", solution.evals.solves);
            println!("  Total steps taken: {}", solution.steps.total());
            println!("  Accepted steps: {}", solution.steps.accepted);
            println!("  Rejected steps: {}", solution.steps.rejected);
        }
        Err(e) => {
            eprintln!("An error occurred: {:?}", e);
        }
    }
}
