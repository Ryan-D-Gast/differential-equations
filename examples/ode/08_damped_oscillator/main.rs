//! Example 08: Damped Oscillator
//!
//! This example simulates a damped harmonic oscillator with the system:
//! dx/dt = v
//! dv/dt = -b*v - k*x
//!
//! where:
//! - x is position
//! - v is velocity
//! - b is the damping coefficient
//! - k is the spring constant
//!
//! Damped oscillators model real-world systems with energy dissipation, such as shock absorbers,
//! RLC circuits, and building structures under damping. The system behavior depends on the
//! damping ratio, which can lead to overdamped, critically damped, or underdamped motion.
//!
//! This example showcases:
//! - Zero-crossing detection with the crossing() method
//! - Using high-precision DOPRI5 integration
//! - Detailed solution statistics reporting

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

/// Damped Harmonic Oscillator ODE
struct DampedOscillator {
    damping: f64,         // Damping coefficient
    spring_constant: f64, // Spring constant
}

impl ODE<f64, SVector<f64, 2>> for DampedOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        // Pure function, no state updates
        dydt[0] = y[1];
        dydt[1] = -self.damping * y[1] - self.spring_constant * y[0];
    }
}

fn main() {
    // Initialize the method
    let mut method = ExplicitRungeKutta::dopri5().rtol(1e-8).atol(1e-8);

    // Define the ode parameters
    let damping = 0.5;
    let spring_constant = 1.0;
    let ode = DampedOscillator {
        damping,
        spring_constant,
    };

    // Define the initial conditions
    let y0 = vector![1.0, 0.0]; // Initial position and velocity
    let t0 = 0.0;
    let tf = 20.0;

    // Create the ODEProblem
    let damped_oscillator_problem = ODEProblem::new(ode, t0, tf, y0);

    // Solve the ODEProblem
    match damped_oscillator_problem
        .crossing(0, 0.0, CrossingDirection::Both)
        .solve(&mut method)
    {
        Ok(solution) => {
            println!("Solution:");
            println!("Time, Position, Velocity");
            for (t, y) in solution.iter() {
                println!("{:.4}, {:.4}, {:.4}", t, y[0], y[1]);
            }

            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {}", solution.steps);
            println!("Rejected Steps: {}", solution.rejected_steps);
            println!("Accepted Steps: {}", solution.accepted_steps);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
