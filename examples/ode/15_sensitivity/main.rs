//! Example 15: Parameter Sensitivity Analysis
//!
//! This example models a Lotka-Volterra (Predator-Prey) system:
//! dx/dt = alpha * x - beta * x * y
//! dy/dt = delta * x * y - gamma * y
//!
//! We compute both the forward parameter sensitivities and the adjoint
//! sensitivities with respect to a terminal cost function (final prey population).
//!
//! The forward sensitivity equations describe how the trajectory y(t)
//! changes with respect to small changes in parameters p.
//! The adjoint method efficiently computes the gradient of a cost functional
//! with respect to all parameters by integrating backward in time.

use differential_equations::ivp::Ivp;
use differential_equations::prelude::*;
use nalgebra::{Vector2, Vector4, vector};
use quill::prelude::*;

struct LotkaVolterra {
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
}

impl ODE<f64, Vector2<f64>> for LotkaVolterra {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        let x = y[0];
        let p = y[1];

        dydt[0] = self.alpha * x - self.beta * x * p;
        dydt[1] = self.delta * x * p - self.gamma * p;
    }

    fn jacobian(&self, _t: f64, y: &Vector2<f64>, jy: &mut Matrix<f64>) {
        let x = y[0];
        let p = y[1];

        jy[(0, 0)] = self.alpha - self.beta * p;
        jy[(0, 1)] = -self.beta * x;
        jy[(1, 0)] = self.delta * p;
        jy[(1, 1)] = self.delta * x - self.gamma;
    }
}

impl VaryParameters<f64, Vector2<f64>, Vector4<f64>> for LotkaVolterra {
    fn parameters(&self) -> Vector4<f64> {
        vector![self.alpha, self.beta, self.delta, self.gamma]
    }

    fn with_parameters(&self, params: &Vector4<f64>) -> Self {
        Self {
            alpha: params[0],
            beta: params[1],
            delta: params[2],
            gamma: params[3],
        }
    }

    fn jacobian_params(&self, _t: f64, y: &Vector2<f64>, jp: &mut Matrix<f64>) {
        let x = y[0];
        let p = y[1];

        // parameters = [alpha, beta, delta, gamma]
        jp[(0, 0)] = x;
        jp[(0, 1)] = -x * p;
        jp[(0, 2)] = 0.0;
        jp[(0, 3)] = 0.0;

        jp[(1, 0)] = 0.0;
        jp[(1, 1)] = 0.0;
        jp[(1, 2)] = x * p;
        jp[(1, 3)] = -p;
    }
}

/// Cost functional measuring the terminal prey population.
struct TerminalPreyCost;

impl AdjointCost<f64, Vector2<f64>, Vector4<f64>> for TerminalPreyCost {
    fn terminal(&self, _tf: f64, yf: &Vector2<f64>, _params: &Vector4<f64>) -> f64 {
        yf[0]
    }

    fn terminal_gradient_y(
        &self,
        _tf: f64,
        _yf: &Vector2<f64>,
        _params: &Vector4<f64>,
        grad_y: &mut Vector2<f64>,
    ) {
        grad_y[0] = 1.0;
        grad_y[1] = 0.0;
    }
}

fn main() -> Result<(), differential_equations::error::Error<f64, Vector2<f64>>> {
    // Initial State: Prey, Predator
    let y0 = vector![10.0, 2.0];
    let t0 = 0.0;
    let tf = 15.0;

    let equation = LotkaVolterra {
        alpha: 1.5,
        beta: 1.0,
        delta: 1.0,
        gamma: 3.0,
    };

    println!("Solving Forward Sensitivity Analysis...");
    let forward_solution = Ivp::ode(&equation, t0, tf, y0)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10))
        .forward_sensitivity()
        .solve()
        .expect("forward sensitivity solve failed");

    println!("Solving Adjoint Sensitivity Analysis...");
    let cost = TerminalPreyCost;
    let forward_method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
    let backward_method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);

    let adjoint_solution = Ivp::ode(&equation, t0, tf, y0)
        .method(forward_method)
        .adjoint_sensitivity(&cost)
        .backward_method(backward_method)
        .solve()?;
    // Print Adjoint Gradients
    println!("\nAdjoint Sensitivities (d(Prey_tf)/dp):");
    println!("d/d(alpha): {:.6}", adjoint_solution.grad_p[0]);
    println!("d/d(beta):  {:.6}", adjoint_solution.grad_p[1]);
    println!("d/d(delta): {:.6}", adjoint_solution.grad_p[2]);
    println!("d/d(gamma): {:.6}", adjoint_solution.grad_p[3]);

    // Check forward sensitivities terminal state to compare
    let fsa_final = forward_solution.y.last().expect("solution has final state");
    println!("\nForward Sensitivities (d(Prey_tf)/dp):");
    println!(
        "d/d(alpha): {:.6} (diff: {:.2e})",
        fsa_final[2],
        (adjoint_solution.grad_p[0] - fsa_final[2]).abs()
    );
    println!(
        "d/d(beta):  {:.6} (diff: {:.2e})",
        fsa_final[4],
        (adjoint_solution.grad_p[1] - fsa_final[4]).abs()
    );
    println!(
        "d/d(delta): {:.6} (diff: {:.2e})",
        fsa_final[6],
        (adjoint_solution.grad_p[2] - fsa_final[6]).abs()
    );
    println!(
        "d/d(gamma): {:.6} (diff: {:.2e})",
        fsa_final[8],
        (adjoint_solution.grad_p[3] - fsa_final[8]).abs()
    );

    println!("\nPlotting results...");

    // Plotting the State and Sensitivities
    Plot::builder()
        .title("Lotka-Volterra with Sensitivities")
        .x_label("Time")
        .y_label("Value")
        .legend(Legend::TopRightInside)
        .data([
            Series::builder()
                .name("Prey")
                .color("Blue")
                .data(
                    forward_solution
                        .iter()
                        .map(|(t, y)| (*t, y[0]))
                        .collect::<Vec<_>>(),
                )
                .build(),
            Series::builder()
                .name("Predator")
                .color("Red")
                .data(
                    forward_solution
                        .iter()
                        .map(|(t, y)| (*t, y[1]))
                        .collect::<Vec<_>>(),
                )
                .build(),
            Series::builder()
                .name("d(Prey)/d(alpha)")
                .color("Green")
                .data(
                    forward_solution
                        .iter()
                        .map(|(t, y)| (*t, y[2])) // y[2] is d(Prey)/d(alpha)
                        .collect::<Vec<_>>(),
                )
                .build(),
        ])
        .build()
        .to_svg("examples/ode/15_sensitivity/sensitivity.svg")
        .expect("Failed to save plot as SVG");

    println!("Saved plot to examples/ode/15_sensitivity/sensitivity.svg");

    Ok(())
}
