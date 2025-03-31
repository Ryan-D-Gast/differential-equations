use differential_equations::ode::*;
use nalgebra::{SMatrix, Matrix2};
use std::f64::consts::PI;

/// Matrix Differential Equation ODE
/// dM/dt = AM - MA (matrix commutator)
struct MatrixODE {
    omega: f64,
}

impl ODE<f64, 2, 2> for MatrixODE {
    fn diff(&self, _t: f64, y: &SMatrix<f64, 2, 2>, dydt: &mut SMatrix<f64, 2, 2>) {
        // Create the rotation generator matrix
        let a = SMatrix::<f64, 2, 2>::new(
            // Essentially random values to show the change in the matrix
            0.2, -self.omega,
            -0.2, self.omega
        );
        
        // Matrix differential equation: dM/dt = AM - MA
        *dydt = a * y - y * a;
    }
}

fn main() {
    // Create a solver
    let mut solver = DOP853::new().rtol(1e-6).atol(1e-6);
    
    // Initial condition: rotation matrix at 45 degrees
    let angle = PI / 4.0;
    let y0 = Matrix2::new(
        angle.cos(), -angle.sin(),
        angle.sin(), angle.cos()
    );
    
    // Solve from t=0 to t=10
    let t0 = 0.0;
    let tf = 10.0;
    let ode = MatrixODE { omega: 0.1 };
    
    // Set up and solve the IVP
    let matrix_ivp = IVP::new(ode, t0, tf, y0);
    let result = matrix_ivp.dense(100).solve(&mut solver);
    
    match result {
        Ok(solution) => {
            println!("Solution at selected points:");
            for (i, (t, y)) in solution.iter().enumerate() {
                if i % 10 == 0 {  // Print every 10th point to keep output manageable
                    println!("t = {:.2}", t);
                    println!("[{:.4}, {:.4}]", y[(0,0)], y[(0,1)]);
                    println!("[{:.4}, {:.4}]", y[(1,0)], y[(1,1)]);
                }
            }
            
            println!("Steps: {}", solution.steps);
            println!("Function evaluations: {}", solution.evals);
        },
        Err(e) => println!(
            "Error solving the IVP: {:?}", e
        ),
    }
}
