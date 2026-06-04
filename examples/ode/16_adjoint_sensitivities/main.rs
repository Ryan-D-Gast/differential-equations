//! Example 16: Adjoint Sensitivity Analysis
//!
//! This example demonstrates how to perform Adjoint Sensitivity Analysis (ASA)
//! using the backward-integration capability of the solvers via the structured
//! `AdjointOde` API.

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

// 1. The Forward Problem
struct ForwardOde {
    p: SVector<f64, 2>,
}

impl ODE<f64, SVector<f64, 2>> for ForwardOde {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = -self.p[0] * y[0] + self.p[1] * y[1];
        dydt[1] = self.p[0] * y[0] - self.p[1] * y[1];
    }
}

impl ParametrizedODE<f64, SVector<f64, 2>, SVector<f64, 2>> for ForwardOde {
    fn parameters(&self) -> SVector<f64, 2> {
        self.p
    }

    fn jacobian_p(&self, _t: f64, y: &SVector<f64, 2>, j: &mut Matrix<f64>) {
        // df/dp
        // df_0/dp_0 = -y_0
        // df_0/dp_1 = y_1
        // df_1/dp_0 = y_0
        // df_1/dp_1 = -y_1
        j[(0, 0)] = -y[0];
        j[(0, 1)] = y[1];
        j[(1, 0)] = y[0];
        j[(1, 1)] = -y[1];
    }
}

fn main() {
    let p = vector![0.1, 0.2];
    let t0 = 0.0;
    let tf = 10.0;
    let y0 = vector![1.0, 0.0];

    // 1. Solve the forward problem with dense output
    let forward_ode = ForwardOde { p };
    let forward_solution = IVP::ode(&forward_ode, t0, tf, y0)
        .dense(10) // high density for accurate interpolation
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    // 2. Solve the adjoint problem backwards using the new builder method
    // Initial condition for backward pass: lambda(T) = dg/dy(T)^T, mu(T) = 0
    let y_final = forward_solution.y.last().unwrap();
    let dg_dy_final = vector![0.0, y_final[1] - 0.5];
    let adjoint_y0 = vector![dg_dy_final[0], dg_dy_final[1], 0.0, 0.0];

    // Integrate backwards
    let adjoint_solution = forward_solution
        .adjoint_sensitivity(&forward_ode, adjoint_y0)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let final_adjoint_state = adjoint_solution.y.last().unwrap();
    let gradient = vector![final_adjoint_state[2], final_adjoint_state[3]];

    println!("Adjoint Sensitivity Analysis");
    println!("============================");
    println!("Computed Gradient w.r.t parameters: {:?}", gradient);

    // Double-check with central finite differences.
    let epsilon = 1e-6;

    let cost = |sol: &Solution<f64, SVector<f64, 2>>| {
        let y_final = sol.y.last().unwrap();
        0.5 * (y_final[1] - 0.5).powi(2)
    };

    let solve_cost = |p_test: SVector<f64, 2>| {
        let sol = IVP::ode(&ForwardOde { p: p_test }, t0, tf, y0)
            .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
            .solve()
            .unwrap();
        cost(&sol)
    };

    let fd_gradient = vector![
        (solve_cost(vector![p[0] + epsilon, p[1]]) - solve_cost(vector![p[0] - epsilon, p[1]]))
            / (2.0 * epsilon),
        (solve_cost(vector![p[0], p[1] + epsilon]) - solve_cost(vector![p[0], p[1] - epsilon]))
            / (2.0 * epsilon),
    ];

    println!("Finite Difference Gradient: {:?}", fd_gradient);
    println!(
        "Gradient error:              {:?}",
        (gradient - fd_gradient).abs()
    );
}
