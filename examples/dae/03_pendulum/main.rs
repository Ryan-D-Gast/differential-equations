//! Example 03: Constrained Pendulum (index-2 DAE)
//!
//! This example demonstrates a constrained pendulum written as a DAE with a
//! Lagrange multiplier. The system variables are [x, y, vx, vy, lambda]
//! where (x,y) is the Cartesian position of the mass, (vx,vy) the velocity,
//! and lambda the Lagrange multiplier enforcing the holonomic constraint
//! x^2 + y^2 = L^2.  The DAE is arranged so that the last equation is the
//! algebraic constraint (mass matrix row = 0) and the dynamics contain the
//! multiplier.  This formulation is a classic index-2 test problem for DAE
//! solvers. Note how during solver setup, we specify the index-2 equations
//! to ensure the solver can handle the algebraic constraints properly.

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use quill::prelude::*;

/// Constrained pendulum model (cartesian coordinates)
struct Pendulum {
    g: f64,
    l: f64,
}

impl Pendulum {
    fn new(g: f64, l: f64) -> Self {
        Self { g, l }
    }
}

// State vector: [x, y, vx, vy, lambda]
impl DAE<f64, SVector<f64, 5>> for Pendulum {
    fn diff(&self, _t: f64, y: &SVector<f64, 5>, f: &mut SVector<f64, 5>) {
        let x = y[0];
        let yy = y[1];
        let vx = y[2];
        let vy = y[3];
        let lambda = y[4];

        // kinematic equations
        f[0] = vx;
        f[1] = vy;

        // dynamics include the Lagrange multiplier (algebraic unknown)
        f[2] = -lambda * x;
        f[3] = -lambda * yy - self.g;

        // algebraic constraint: position must remain on circle of radius L
        f[4] = x * x + yy * yy - self.l * self.l;
    }

    fn mass(&self, m: &mut Matrix<f64>) {
        // Differential eqns have mass 1, constraint is algebraic (mass 0)
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 1.0;
        m[(2, 2)] = 1.0;
        m[(3, 3)] = 1.0;
        m[(4, 4)] = 0.0;
    }

    fn jacobian(&self, _t: f64, y: &SVector<f64, 5>, j: &mut Matrix<f64>) {
        let x = y[0];
        let yy = y[1];
        let lambda = y[4];

        // Row 0: dx/dt = vx
        j[(0, 0)] = 0.0;
        j[(0, 1)] = 0.0;
        j[(0, 2)] = 1.0;
        j[(0, 3)] = 0.0;
        j[(0, 4)] = 0.0;

        // Row 1: dy/dt = vy
        j[(1, 0)] = 0.0;
        j[(1, 1)] = 0.0;
        j[(1, 2)] = 0.0;
        j[(1, 3)] = 1.0;
        j[(1, 4)] = 0.0;

        // Row 2: dvx/dt = -lambda * x  => d/dx = -lambda, d/dlambda = -x
        j[(2, 0)] = -lambda;
        j[(2, 1)] = 0.0;
        j[(2, 2)] = 0.0;
        j[(2, 3)] = 0.0;
        j[(2, 4)] = -x;

        // Row 3: dvy/dt = -lambda * y - g
        j[(3, 0)] = 0.0;
        j[(3, 1)] = -lambda;
        j[(3, 2)] = 0.0;
        j[(3, 3)] = 0.0;
        j[(3, 4)] = -yy;

        // Row 4: constraint d/dy of x^2 + y^2 - L^2
        j[(4, 0)] = 2.0 * x;
        j[(4, 1)] = 2.0 * yy;
        j[(4, 2)] = 0.0;
        j[(4, 3)] = 0.0;
        j[(4, 4)] = 0.0;
    }
}

fn main() {
    // Choose solver (stiff implicit solver suitable for higher-index DAE)
    let mut method = ImplicitRungeKutta::radau5()
        .rtol(1e-8)
        .atol([1e-10, 1e-10, 1e-10, 1e-10, 1e-12])
        // Required for index-2 (or index-3) dae problems otherwise step size will converge to zero.
        .index2_equations_idxs(vec![4]);

    let g = 9.81;
    let l = 1.0;
    let model = Pendulum::new(g, l);

    // Initial angle (radians) and zero velocity
    let theta0 = std::f64::consts::FRAC_PI_4; // 45 degrees
    let x0 = l * theta0.sin();
    let y0 = -l * theta0.cos();
    let vx0 = 0.0;
    let vy0 = 0.0;

    // Compute consistent initial lambda from second derivative of constraint:
    // x*ax + y*ay + vx^2 + vy^2 = 0, with ax = -lambda*x, ay = -lambda*y - g
    // => -lambda*(x^2 + y^2) - y*g + vx^2 + vy^2 = 0  => lambda = -( - y*g + vx^2 + vy^2)/(l^2)
    let lambda0 = -(-y0 * g + vx0 * vx0 + vy0 * vy0) / (l * l);
    let y0 = vector![x0, y0, vx0, vy0, lambda0];
    let t0 = 0.0;
    let tf = 10.0;
    let problem = DAEProblem::new(model, t0, tf, y0);
    match problem.even(0.05).solve(&mut method) {
        Ok(solution) => {
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());

            println!("\nConstrained pendulum solution (x, y):");
            println!("{:>8}  {:>12}  {:>12}  {:>12}", "Time", "x", "y", "lambda");
            for (t, y) in solution.iter() {
                println!("{:8.4}  {:12.8}  {:12.8}  {:12.8}", t, y[0], y[1], y[4]);
            }

            // Plot x and y vs time
            let s1: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[0])).collect();
            let s2: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[1])).collect();

            Plot::builder()
                .title("Constrained pendulum: x and y vs time")
                .x_label("t")
                .y_label("x, y")
                .legend(Legend::TopLeftInside)
                .data([
                    Series::builder()
                        .name("x   ")
                        .color("Blue")
                        .data(s1)
                        .build(),
                    Series::builder()
                        .name("y   ")
                        .color("Green")
                        .data(s2)
                        .build(),
                ])
                .build()
                .to_svg("examples/dae/03_pendulum/pendulum.svg")
                .expect("Failed to save pendulum plot as SVG");
        }
        Err(e) => panic!("Error solving pendulum DAE: {:?}", e),
    }
}
