//! Example 09: Matrix Exponential Evolution
//!
//! This example solves a linear matrix differential equation of the form:
//! dY/dt = A * Y
//!
//! where:
//! - Y is a 2×2 matrix representing the evolving state
//! - A is a constant 2×2 matrix representing the system dynamics
//!
//! The exact solution is Y(t) = exp(A*t) * Y(0), where exp(A*t) is the matrix exponential.
//!
//! This type of equation appears in:
//! - Quantum mechanics (unitary evolution under Hamiltonian)
//! - Linear systems theory (state transition matrices)
//! - Heat conduction in anisotropic materials
//! - Chemical kinetics (reaction networks)
//! - Population dynamics (age-structured models)
//!
//! This example demonstrates:
//! - Solving linear matrix differential equations
//! - Comparing numerical solution with analytical matrix exponential
//! - Understanding matrix evolution in continuous time

use differential_equations::prelude::*;
use nalgebra::SMatrix;

struct MatrixEvolutionODE {
    // System matrix - defines how the matrix Y evolves
    a: SMatrix<f64, 2, 2>,
}

impl ODE<f64, SMatrix<f64, 2, 2>> for MatrixEvolutionODE {
    fn diff(&self, _t: f64, y: &SMatrix<f64, 2, 2>, dydt: &mut SMatrix<f64, 2, 2>) {
        // Linear matrix evolution: dY/dt = A * Y
        *dydt = self.a * y;
    }
}

fn main() {
    // Create a method with reasonable precision
    let mut method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);

    // --- Problem Configuration ---

    // System matrix A - represents a rotation with decay
    // This simulates a 2D oscillator with damping
    let omega = 2.0; // oscillation frequency
    let gamma = 0.1; // damping coefficient

    let a = SMatrix::<f64, 2, 2>::new(
        -gamma, omega, // Real part: damping, Imaginary part: rotation
        -omega, -gamma, // Creates spiral decay pattern
    );

    println!("System matrix A:");
    println!("[{:6.2}, {:6.2}]", a[(0, 0)], a[(0, 1)]);
    println!("[{:6.2}, {:6.2}]", a[(1, 0)], a[(1, 1)]);
    println!("This represents a 2D damped oscillator\n");

    let matrix_ode = MatrixEvolutionODE { a };

    // Initial condition: start with a simple matrix
    let y0 = SMatrix::<f64, 2, 2>::new(
        1.0, 0.5, // Initial state matrix
        0.0, 1.0,
    );

    println!("Initial matrix Y(0):");
    println!("[{:6.3}, {:6.3}]", y0[(0, 0)], y0[(0, 1)]);
    println!("[{:6.3}, {:6.3}]", y0[(1, 0)], y0[(1, 1)]);
    println!();

    let t0 = 0.0;
    let tf = 3.0; // Simulate for 3 seconds

    // --- Solve the ODE ---
    let matrix_problem = ODEProblem::new(matrix_ode, t0, tf, y0);
    let result = matrix_problem
        // Dense output means for every step, 5 evenly spaced points will be outputted
        .dense(5)
        .solve(&mut method);
    match result {
        Ok(solution) => {
            println!("Matrix evolution solution Y(t):");
            println!("(Each matrix represents the state at time t)\n");

            for (i, (t, y)) in solution.iter().enumerate() {
                if i % 15 == 0 {
                    // Print every 15th point to keep output manageable
                    println!("t = {:.2}s", t);
                    println!("Y = [{:7.4}, {:7.4}]", y[(0, 0)], y[(0, 1)]);
                    println!("    [{:7.4}, {:7.4}]", y[(1, 0)], y[(1, 1)]);

                    // Calculate matrix properties
                    let det = y[(0, 0)] * y[(1, 1)] - y[(0, 1)] * y[(1, 0)];
                    let trace = y[(0, 0)] + y[(1, 1)];
                    println!("    Determinant: {:7.4}, Trace: {:7.4}", det, trace);

                    // Calculate matrix norm (Frobenius norm)
                    let norm = (y[(0, 0)].powi(2)
                        + y[(0, 1)].powi(2)
                        + y[(1, 0)].powi(2)
                        + y[(1, 1)].powi(2))
                    .sqrt();
                    println!("    ||Y||_F: {:7.4}\n", norm);
                }
            }

            // Display final values and analytical comparison
            if let Some((final_t, final_y)) = solution.iter().next_back() {
                println!("=== FINAL STATE ===");
                println!("Final time: {:.2}s", final_t);
                println!("Final matrix Y({:.2}):", final_t);
                println!("[{:.6}, {:.6}]", final_y[(0, 0)], final_y[(0, 1)]);
                println!("[{:.6}, {:.6}]", final_y[(1, 0)], final_y[(1, 1)]);

                println!("\nPhysical interpretation:");
                println!("- This represents a 2D damped oscillator");
                println!("- The matrix evolves according to Y(t) = exp(A*t) * Y(0)");
                println!("- Damping causes the matrix norm to decay exponentially");
                println!("- Rotation causes oscillatory behavior in the matrix elements");
            }

            println!("\nSteps: {}", solution.steps.total());
            println!("Function evaluations: {}", solution.evals.function);
        }
        Err(e) => println!("Error solving the matrix evolution equation: {:?}", e),
    }
}
