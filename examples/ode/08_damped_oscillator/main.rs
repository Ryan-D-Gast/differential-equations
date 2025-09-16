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

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

struct DampedOscillator {
    damping: f64,
    spring_constant: f64,
}

impl ODE<f64, SVector<f64, 2>> for DampedOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.damping * y[1] - self.spring_constant * y[0];
    }
}

fn main() {
    // --- Problem Configuration ---
    let damping = 0.5;
    let spring_constant = 1.0;
    let ode = DampedOscillator {
        damping,
        spring_constant,
    };
    let y0 = vector![1.0, 0.0];
    let t0 = 0.0;
    let tf = 20.0;
    let damped_oscillator_problem = ODEProblem::new(&ode, t0, tf, y0);

    // --- Solve the ODE ---
    let mut method = ExplicitRungeKutta::dopri5().rtol(1e-8).atol(1e-8);
    match damped_oscillator_problem
        // Detect zero-crossing of the component of index 0 (x=position) at the value 0.0 from both positive and negative directions
        .crossing(0, 0.0, CrossingDirection::Both)
        .solve(&mut method)
    {
        Ok(solution) => {
            println!("Solution:");
            println!("Time, Position, Velocity");
            for (t, y) in solution.iter() {
                println!("{:.4}, {:.4}, {:.4}", t, y[0], y[1]);
            }

            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
