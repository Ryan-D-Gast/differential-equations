//! # Example 1: Mackey-Glass Delay Differential Equation
//!
//! This example demonstrates the solution of the Mackey-Glass equation,
//! a classic example of a delay differential equation known for
//! exhibiting chaotic behavior for certain parameter values.
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
use quill::prelude::*;

struct MackeyGlass {
    beta: f64,
    gamma: f64,
    n: f64,
    tau: f64,
}

// Generic "L" is the number of lags, here we have 1 lag tau
impl DDE<1> for MackeyGlass {
    fn diff(&self, _t: f64, y: &f64, yd: &[f64; 1], dydt: &mut f64) {
        *dydt = (self.beta * yd[0]) / (1.0 + yd[0].powf(self.n)) - self.gamma * *y;
    }

    // Define a constant delay for the model of tau
    fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
        lags[0] = self.tau;
    }
}

fn main() {
    // --- Solver Configuration ---
    let mut solver = ExplicitRungeKutta::rkv878e().max_delay(20.0); // Discard history older than 20.0 seconds to save memory

    // --- Problem Definition ---
    let dde = MackeyGlass {
        beta: 0.2,
        gamma: 0.1,
        n: 10.0,
        tau: 20.0,
    };

    // Define initial conditions
    let t0 = 0.0;
    let tf = 200.0;
    let y0 = 0.1;

    // Define the initial history function phi(t) for t <= t0
    // Often a constant history is used matching the initial condition
    let phi = |_t: f64| -> f64 { y0 };

    // Define the DDE problem
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
            println!("Function evaluations: {}", solution.evals.function);
            println!("Solver steps: {}", solution.steps.total());
            println!("Accepted steps: {}", solution.steps.accepted);
            println!("Rejected steps: {}", solution.steps.rejected);
            println!("Number of output points: {}", solution.t.len());

            // Plot the solution using the quill library
            Plot::builder()
                .title("Mackey-Glass Delay Differential Equation")
                .x_label("Time (t)")
                .y_label("y(t)")
                .data([Series::builder()
                    .name("Mackey-Glass Solution")
                    .color("Blue")
                    .data(solution.iter().map(|(t, y)| (*t, *y)).collect::<Vec<_>>())
                    .build()])
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
