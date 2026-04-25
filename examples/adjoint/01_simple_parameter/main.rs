//! Example 01: Simple Parameter Sensitivity
//!
//! This example demonstrates using Adjoint Sensitivity Analysis to compute
//! the gradients of a cost function with respect to the initial state and parameters.

use differential_equations::adjoint::cost::CostFunction;
use differential_equations::adjoint::solve::solve_adjoint;
use differential_equations::adjoint::system::ParameterizedODE;
use differential_equations::ode::ODE;
use differential_equations::prelude::*;
use differential_equations::solout::DefaultSolout;
use nalgebra::{Vector1, vector};

// System: dy/dt = -p * y
// Exact solution: y(t) = y0 * e^(-p * t)
struct DecayModel;

impl ODE<f64, Vector1<f64>> for DecayModel {
    fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
        dydt[0] = -y[0]; // Without parameters
    }
}

// Parameterized System extension
impl ParameterizedODE<f64, Vector1<f64>, Vector1<f64>> for DecayModel {
    fn diff_p(&self, _t: f64, y: &Vector1<f64>, p: &Vector1<f64>, dydt: &mut Vector1<f64>) {
        dydt[0] = -p[0] * y[0];
    }
}

// Cost function: J(y, p) = y(t_f)
struct TerminalCost;

impl CostFunction<f64, Vector1<f64>, Vector1<f64>> for TerminalCost {
    fn discrete(&self, t: f64, y: &Vector1<f64>, _p: &Vector1<f64>) -> f64 {
        if t == 2.0 {
            y[0] // Cost is exactly the terminal state
        } else {
            0.0
        }
    }
}

fn main() {
    let ode = DecayModel;
    let cost = TerminalCost;

    let t0 = 0.0;
    let tf = 2.0;
    let y0 = vector![1.0];
    let p = vector![0.5];

    let mut forward_solver = ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-10);
    let mut backward_solver = ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-10);
    let mut backward_solout = DefaultSolout::new();

    let (_forward_sol, adjoint_sol) = solve_adjoint(
        &mut forward_solver,
        &mut backward_solver,
        &ode,
        &cost,
        t0,
        tf,
        &y0,
        &p,
        &mut backward_solout,
    )
    .unwrap();

    let adj_final = adjoint_sol.y.last().unwrap();

    println!("Gradients evaluated at t0={}:", t0);
    println!("dJ/dy0 = lambda = {}", adj_final.lambda[0]);
    println!("dJ/dp = mu = {}", adj_final.mu[0]);

    // Expected analytical derivatives:
    // y(tf) = y0 * exp(-p * tf)
    // dJ/dy0 = exp(-p * tf) = exp(-0.5 * 2.0) = exp(-1) ≈ 0.367879
    // dJ/dp = y0 * (-tf) * exp(-p * tf) = 1.0 * (-2.0) * exp(-1) ≈ -0.735758

    println!("\nExpected analytical values:");
    println!("dJ/dy0 = {}", (-1.0_f64).exp());
    println!("dJ/dp = {}", -2.0 * (-1.0_f64).exp());
}
