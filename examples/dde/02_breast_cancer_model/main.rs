//! # Example 2: Breast Cancer Model with Delay
//!
//! This example demonstrates the solution of a delay differential equation model
//! describing the dynamics of breast cancer cells under treatment, considering
//! different cell populations (proliferating, quiescent, and resistant).
//!
//! The model equations are:
//! du₁/dt = (v₀ / (1 + β₀ * u₃(t-τ)²)) * (p₀ - q₀) * u₁ - d₀ * u₁
//! du₂/dt = (v₀ / (1 + β₀ * u₃(t-τ)²)) * (1 - p₀ + q₀) * u₁ + (v₁ / (1 + β₁ * u₃(t-τ)²)) * (p₁ - q₁) * u₂ - d₁ * u₂
//! du₃/dt = (v₁ / (1 + β₁ * u₃(t-τ)²)) * (1 - p₁ + q₁) * u₂ - d₂ * u₃
//!
//! Source: [https://www.nature.com/articles/srep02473]
//!
//! where:
//! - u₁, u₂, u₃ represent different cell populations.
//! - u₃(t-τ) is the value of the third component at a delayed time t - τ.
//! - p₀, q₀, v₀, d₀, p₁, q₁, v₁, d₁, d₂, β₀, β₁, τ are model parameters.

use differential_equations::prelude::*;
use nalgebra::Vector3;
use quill::prelude::*;

struct BreastCancerModel {
    p0: f64,
    q0: f64,
    v0: f64,
    d0: f64,
    p1: f64,
    q1: f64,
    v1: f64,
    d1: f64,
    d2: f64,
    beta0: f64,
    beta1: f64,
    tau: f64,
}

impl DDE<1, f64, Vector3<f64>> for BreastCancerModel {
    fn diff(&self, _t: f64, u: &Vector3<f64>, ud: &[Vector3<f64>; 1], dudt: &mut Vector3<f64>) {
        let hist3 = ud[0][2];

        let term0_common = self.v0 / (1.0 + self.beta0 * hist3.powi(2));
        let term1_common = self.v1 / (1.0 + self.beta1 * hist3.powi(2));

        dudt[0] = term0_common * (self.p0 - self.q0) * u[0] - self.d0 * u[0];
        dudt[1] = term0_common * (1.0 - self.p0 + self.q0) * u[0]
            + term1_common * (self.p1 - self.q1) * u[1]
            - self.d1 * u[1];
        dudt[2] = term1_common * (1.0 - self.p1 + self.q1) * u[1] - self.d2 * u[2];
    }

    fn lags(&self, _t: f64, _y: &Vector3<f64>, lags: &mut [f64; 1]) {
        lags[0] = self.tau;
    }
}

fn main() {
    // --- Solver Configuration ---
    let mut solver = ExplicitRungeKutta::rkv766e()
        .max_delay(1.0);

    // --- Problem Definition ---
    let tau = 1.0;
    let dde = BreastCancerModel {
        p0: 0.2,
        q0: 0.3,
        v0: 1.0,
        d0: 5.0,
        p1: 0.2,
        q1: 0.3,
        v1: 1.0,
        d1: 1.0,
        d2: 1.0,
        beta0: 1.0,
        beta1: 1.0,
        tau,
    };
    let t0 = 0.0;
    let tf = 10.0;
    let y0 = Vector3::new(1.0, 1.0, 1.0);
    let phi = |_t: f64| -> Vector3<f64> { y0 };
    let problem = DDEProblem::new(dde, t0, tf, y0, phi);

    // --- Solve the Problem ---
    println!(
        "Solving Breast Cancer Model (tau={}) from t={} to t={}...",
        tau, t0, tf
    );
    match problem.even(0.1).solve(&mut solver) {
        Ok(solution) => {
            println!("Solver finished with status: {:?}", solution.status);

            // Print statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Solver steps: {}", solution.steps.total());
            println!("Accepted steps: {}", solution.steps.accepted);
            println!("Rejected steps: {}", solution.steps.rejected);
            println!("Number of output points: {}", solution.t.len());

            // Print the solution
            for (t, u) in solution.iter() {
                println!(
                    "t: {:.4}, u1: {:.4}, u2: {:.4}, u3: {:.4}",
                    t, u[0], u[1], u[2]
                );
            }

            // Plotting
            Plot::builder()
                .title("Breast Cancer Model with Delay")
                .x_label("Time (t)")
                .y_label("Cell Population")
                .legend(Legend::TopRightInside)
                .data([
                    Series::builder()
                        .name("Proliferating Cells (u1)")
                        .color("Blue")
                        .data(solution.iter().map(|(t, u)| (*t, u[0])).collect::<Vec<_>>())
                        .build(),
                    Series::builder()
                        .name("Quiescent Cells (u2)")
                        .color("Red")
                        .data(solution.iter().map(|(t, u)| (*t, u[1])).collect::<Vec<_>>())
                        .build(),
                    Series::builder()
                        .name("Resistant Cells (u3)")
                        .color("Green")
                        .data(solution.iter().map(|(t, u)| (*t, u[2])).collect::<Vec<_>>())
                        .build(),
                ])
                .build()
                .to_svg("examples/dde/02_breast_cancer_model/breast_cancer_model.svg")
                .expect("Failed to save plot as SVG");
        }
        Err(e) => {
            eprintln!("Error solving DDE: {:?}", e);
            panic!("Solver failed.");
        }
    };
}
