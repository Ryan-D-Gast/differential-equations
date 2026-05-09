//! Example 18: Telemetry and Observability
//!
//! This example demonstrates how to use the `tracing` framework with the solvers.
//! Solvers emit telemetry about step rejections, order changes, and Newton iterations,
//! allowing for deep insights into solver behavior and debugging.

use differential_equations::prelude::*;
use tracing_subscriber::{EnvFilter, fmt};

struct StiffODE;

impl ODE for StiffODE {
    fn diff(&self, t: f64, y: &f64, dydt: &mut f64) {
        // A moderately stiff ODE: y' = -50 * (y - cos(t)) - sin(t)
        *dydt = -50.0 * (*y - t.cos()) - t.sin();
    }
}

fn main() {
    // Set up `tracing` to output solver telemetry to standard output
    // Run this example with `RUST_LOG=trace cargo run --example ode_18_telemetry` to see the output
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    tracing::info!("Starting integration of stiff ODE to demonstrate telemetry.");

    let y0 = 0.0;
    let t0 = 0.0;
    let tf = 2.0;
    let ode = StiffODE;

    // We use BDF, an adaptive multi-step solver, which can change order and reject steps
    let solution = match IVP::ode(&ode, t0, tf, y0)
        .method(BDF::adaptive().rtol(1e-4).atol(1e-4))
        .solve()
    {
        Ok(solution) => solution,
        Err(e) => panic!("Error: {:?}", e),
    };

    tracing::info!("Integration complete.");
    tracing::info!("Steps accepted: {}", solution.steps.accepted);
    tracing::info!("Steps rejected: {}", solution.steps.rejected);
}
