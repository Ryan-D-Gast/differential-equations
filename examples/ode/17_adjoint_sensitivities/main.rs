//! Example 17: Adjoint Sensitivity Analysis
//!
//! This example demonstrates how to perform Adjoint Sensitivity Analysis (ASA)
//! using the backward-integration capability of the solvers.
//!
//! ASA calculates the gradient of a cost function with respect to parameters
//! by solving an adjoint ODE backward in time.

use differential_equations::ivp::Ivp;
use differential_equations::prelude::*;
use nalgebra::{Matrix2, SVector, vector};

// 1. The Forward Problem
// Consider the problem:
// dy_0/dt = -p_0 * y_0 + p_1 * y_1
// dy_1/dt = p_0 * y_0 - p_1 * y_1
// With y(0) = [1.0, 0.0], p = [0.1, 0.2]
struct ForwardOde {
    p: SVector<f64, 2>,
}

impl ODE<f64, SVector<f64, 2>> for ForwardOde {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = -self.p[0] * y[0] + self.p[1] * y[1];
        dydt[1] = self.p[0] * y[0] - self.p[1] * y[1];
    }
}

// 2. The Adjoint Problem
// We want to minimize the cost function: G = 0.5 * (y_1(T) - 0.5)^2
// (Note: This is a discrete cost function at the final time T)
//
// Let g(y, p) = 0.5 * (y_1(T) - 0.5)^2
// dg/dy(T) = [0, y_1(T) - 0.5]
//
// The forward Jacobian is:
// J_y = [-p_0, p_1]
//       [ p_0,-p_1]
//
// J_p = [-y_0, y_1]
//       [ y_0,-y_1]
//
// The adjoint variables lambda satisfy:
// d lambda / dt = -J_y^T lambda
// with final condition: lambda(T) = dg/dy(T)^T
//
// The parameter-gradient accumulator mu satisfies:
// d mu / dt = -J_p^T lambda
// with final condition: mu(T) = 0

struct AdjointOde {
    p: SVector<f64, 2>,
    forward_solution: Solution<f64, SVector<f64, 2>>,
}

// The state for the adjoint problem is composed of lambda and mu
// Let's use a 4D vector: [lambda_0, lambda_1, mu_0, mu_1]
impl ODE<f64, SVector<f64, 4>> for AdjointOde {
    fn diff(&self, t: f64, adjoint_state: &SVector<f64, 4>, dydt: &mut SVector<f64, 4>) {
        let y = self.interpolate_forward(t);

        let lambda = SVector::<f64, 2>::new(adjoint_state[0], adjoint_state[1]);

        // J_y^T
        let j_y_t = Matrix2::new(-self.p[0], self.p[0], self.p[1], -self.p[1]);

        // J_p^T
        let j_p_t = Matrix2::new(-y[0], y[0], y[1], -y[1]);

        // d lambda / dt = -J_y^T lambda
        let d_lambda_dt = -j_y_t * lambda;

        // d mu / dt = -J_p^T lambda
        let d_mu_dt = -j_p_t * lambda;

        dydt[0] = d_lambda_dt[0];
        dydt[1] = d_lambda_dt[1];
        dydt[2] = d_mu_dt[0];
        dydt[3] = d_mu_dt[1];
    }
}

impl AdjointOde {
    fn interpolate_forward(&self, t: f64) -> SVector<f64, 2> {
        let times = &self.forward_solution.t;
        let states = &self.forward_solution.y;

        if times.is_empty() {
            return vector![0.0, 0.0];
        }

        if t <= times[0] {
            return states[0];
        }

        if t >= *times.last().unwrap() {
            return *states.last().unwrap();
        }

        let upper = times.partition_point(|ti| *ti < t);
        let lower = upper - 1;
        let s = (t - times[lower]) / (times[upper] - times[lower]);
        states[lower] * (1.0 - s) + states[upper] * s
    }
}

fn main() {
    let p = vector![0.1, 0.2];
    let t0 = 0.0;
    let tf = 10.0;
    let y0 = vector![1.0, 0.0];

    // 1. Solve the forward problem with dense output
    let forward_ode = ForwardOde { p };
    let forward_solution = Ivp::ode(&forward_ode, t0, tf, y0)
        .dense(10) // high density for accurate linear interpolation
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    // 2. Solve the adjoint problem backwards
    let adjoint_ode = AdjointOde {
        p,
        forward_solution,
    };

    // Initial condition for backward pass: lambda(T) = dg/dy(T)^T, mu(T) = 0
    let y_final = adjoint_ode.forward_solution.y.last().unwrap();
    let dg_dy_final = vector![0.0, y_final[1] - 0.5];
    let adjoint_y0 = vector![dg_dy_final[0], dg_dy_final[1], 0.0, 0.0];

    // Integrate backwards from tf to t0
    let adjoint_solution = Ivp::ode(&adjoint_ode, tf, t0, adjoint_y0)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let final_adjoint_state = adjoint_solution.y.last().unwrap();
    let gradient = vector![final_adjoint_state[2], final_adjoint_state[3]];

    println!("Adjoint Sensitivity Analysis");
    println!("============================");
    // The backward integration returns the accumulator at t0. With the sign
    // convention above, mu(t0) is the gradient of the terminal cost with
    // respect to the parameters.
    println!("Computed Gradient w.r.t parameters: {:?}", gradient);

    // Double-check with central finite differences.
    let epsilon = 1e-6;

    let cost = |sol: &Solution<f64, SVector<f64, 2>>| {
        let y_final = sol.y.last().unwrap();
        0.5 * (y_final[1] - 0.5).powi(2)
    };

    let solve_cost = |p_test: SVector<f64, 2>| {
        let sol = Ivp::ode(&ForwardOde { p: p_test }, t0, tf, y0)
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
