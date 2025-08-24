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
            println!(
                "  Total LU decompositions: {}",
                solution.evals.decompositions
            );
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

/* Fortran Radau5 output
PS C:\Users\Ryan\Desktop\Code\Rust\differential-equations\tools\radau_vanderpol_test> .\radau_vanderpol.exe
 X = 0.00    Y =  0.2000000000E+01 -0.6600000000E+00    NSTEP =   0
 X = 0.20    Y =  0.1858198964E+01 -0.7574791034E+00    NSTEP =  10
 X = 0.40    Y =  0.1693205231E+01 -0.9069021617E+00    NSTEP =  11
 X = 0.60    Y =  0.1484565763E+01 -0.1233096965E+01    NSTEP =  13
 X = 0.80    Y =  0.1083912895E+01 -0.6196077501E+01    NSTEP =  22
 X = 1.00    Y = -0.1863645642E+01  0.7535323767E+00    NSTEP = 123
 X = 1.20    Y = -0.1699724325E+01  0.8997692747E+00    NSTEP = 124
 X = 1.40    Y = -0.1493375073E+01  0.1213880891E+01    NSTEP = 126
 X = 1.60    Y = -0.1120780441E+01  0.4374794949E+01    NSTEP = 133
 X = 1.80    Y =  0.1869050586E+01 -0.7495753072E+00    NSTEP = 237
 X = 2.00    Y =  0.1706161101E+01 -0.8928074777E+00    NSTEP = 238
 X = 2.00    Y =  0.1706161101E+01 -0.8928074777E+00
       rtol=0.10D-03
 fcn= 2218 jac= 161 step= 275 accpt= 238 rejct=  8 dec= 248 sol=  660
*/
