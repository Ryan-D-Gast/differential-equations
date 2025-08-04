//! Example 09: Matrix Differential Equation
//!
//! This example solves a matrix differential equation of the form:
//! dM/dt = AM - MA
//!
//! where:
//! - M is a 2×2 matrix representing the state
//! - A is a 2×2 matrix representing the system dynamics
//! - The expression AM - MA is the matrix commutator [A,M]
//!
//! Matrix differential equations appear in quantum mechanics, rigid body dynamics,
//! control theory, and many other fields where the evolution of matrix quantities
//! (like rotation, stress tensors, or operators) is important.
//!
//! This example demonstrates:
//! - Using matrix types (SMatrix from nalgebra) as ODE state
//! - Solving higher-dimensional problems with matrix operations
//! - Displaying selected solution points to manage output

use differential_equations::prelude::*;
use nalgebra::{Matrix2, SMatrix};
use std::f64::consts::PI;

struct MatrixODE {
    omega: f64,
}

impl ODE<f64, SMatrix<f64, 2, 2>> for MatrixODE {
    fn diff(&self, _t: f64, y: &SMatrix<f64, 2, 2>, dydt: &mut SMatrix<f64, 2, 2>) {
        // Create the rotation generator matrix
        let a = SMatrix::<f64, 2, 2>::new(
            // Essentially random values to show the change in the matrix
            0.2,
            -self.omega,
            -0.2,
            self.omega,
        );

        // Matrix differential equation: dM/dt = AM - MA
        *dydt = a * y - y * a;
    }
}

fn main() {
    // Create a method
    let mut method = ExplicitRungeKutta::dop853().rtol(1e-6).atol(1e-6);

    // --- Problem Configuration ---
    let angle = PI / 4.0;
    let y0 = Matrix2::new(angle.cos(), -angle.sin(), angle.sin(), angle.cos()); // rotation matrix at 45 degrees
    let t0 = 0.0;
    let tf = 10.0;
    let ode = MatrixODE { omega: 0.1 };

    // --- Solve the ODE ---
    let matrix_problem = ODEProblem::new(ode, t0, tf, y0);
    let result = matrix_problem
        // Dense output means for every step, 5 evenly spaced points will be outputted
        .dense(5)
        .solve(&mut method);
    match result {
        Ok(solution) => {
            println!("Solution at selected points:");
            for (i, (t, y)) in solution.iter().enumerate() {
                if i % 10 == 0 {
                    // Print every 10th point to keep output manageable
                    println!("t = {:.2}", t);
                    println!("[{:.4}, {:.4}]", y[(0, 0)], y[(0, 1)]);
                    println!("[{:.4}, {:.4}]", y[(1, 0)], y[(1, 1)]);
                }
            }

            println!("Steps: {}", solution.steps.total());
            println!("Function evaluations: {}", solution.evals.function);
        }
        Err(e) => println!("Error solving the ODEProblem: {:?}", e),
    }
}
