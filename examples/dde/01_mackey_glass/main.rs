//! # Example 1: Mackey-Glass Delay Differential Equation
//!
//! This example demonstrates the solution of the Mackey-Glass equation, a classic example of a delay differential equation known for exhibiting chaotic behavior for certain parameter values.
//!
//! The equation is:
//! dy/dt = β * y(t - τ) / (1 + y(t - τ)^n) - γ * y(t)
//!
//! where:
//! - y(t) is the variable at time t
//! - y(t - τ) is the variable at a delayed time t - τ
//! - β, γ, n, τ are parameters of the model.
//!
//! This equation was originally proposed as a model for the production of blood cells.

use differential_equations::prelude::*;
use quill::*;

struct MackeyGlass {
    beta: f64,
    gamma: f64,
    n: f64,
    tau: f64,
}

impl DDE<1> for MackeyGlass {
    fn diff(&self, _t: f64, y: &f64, yd: &[f64; 1], dydt: &mut f64) {
        *dydt = (self.beta * yd[0]) / (1.0 + yd[0].powf(self.n)) - self.gamma * *y;
    }

    fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
        // Set the delay to the fixed value tau
        lags[0] = self.tau;
    }
}

fn main() {
    // --- Solver Configuration ---
    let mut solver = ExplicitRungeKutta::rkv878e() // Use the Delay version of rkv878e solver
        .max_delay(20.0); // Set the maximum delay to match the problem's tau so unnecessary history can be discarded as the solver progresses (optional)

    // --- Problem Definition ---
    let dde = MackeyGlass {
        beta: 0.2,
        gamma: 0.1,
        n: 10.0,
        tau: 20.0,
    };

    // Define initial conditions and time span
    let t0 = 0.0;
    let tf = 200.0; // Integrate over a longer time span to observe behavior
    let y0 = 0.1; // Initial value at t=t0

    // Define the initial history function phi(t) for t <= t0
    // Often a constant history is used.
    let phi = |_t: f64| -> f64 {
        y0 // Use the initial value as constant history
    };

    // Create the DDEProblem
    let problem = DDEProblem::new(dde, t0, tf, y0, phi);

    // --- Solve the Problem ---
    println!("Solving Mackey-Glass equation from t={} to t={}...", t0, tf);
    match problem.even(2.0).solve(&mut solver) {
        Ok(solution) => {
            // Print the solution
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y);
            }

            // Print summary statistics
            println!("Function evaluations: {}", solution.evals);
            println!("Solver steps: {}", solution.steps);
            println!("Accepted steps: {}", solution.accepted_steps);
            println!("Rejected steps: {}", solution.rejected_steps);
            println!("Number of output points: {}", solution.t.len());

            // Plot the solution using quill
            Plot::builder()
                .title("Mackey-Glass Delay Differential Equation".to_string())
                .x_label("Time (t)".to_string())
                .y_label("y(t)".to_string())
                .data(vec![
                    Series::builder()
                        .name("Mackey-Glass Solution".to_string())
                        .color("Blue".to_string())
                        .data(
                            solution
                                .iter()
                                .map(|(t, y)| (*t, *y))
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                ])
                .build()
                .to_svg("examples/dde/01_mackey_glass/mackey_glass.svg")
                .expect("Failed to save plot as SVG");
        }
        Err(e) => {
            eprintln!("Error solving DDE: {:?}", e);
            panic!("Solver failed.");
        }
    };
}
