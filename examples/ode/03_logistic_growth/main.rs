//! # Example 3: Logistic Growth Model
//! 
//! This example demonstrates the solution of a logistic growth model ODE:
//! dy/dt = k*y*(1 - y/m)
//!
//! where:
//! - y is the population size
//! - k is the growth rate constant
//! - m is the carrying capacity of the environment
//! - t is time
//! The logistic growth model is used in ecology to describe how populations grow in an environment with limited resources.
//! This example showcases:
//! - Custom ODE implementation with event detection
//! - Using the DOP853 method for high accuracy
//! - Handling events during the solution process
//! - Accessing solution statistics like step counts and evaluations

use differential_equations::prelude::*;

struct LogisticGrowth {
    k: f64,
    m: f64,
}

impl ODE for LogisticGrowth {
    fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.k * y * (1.0 - y / self.m);
    }

    fn event(&self, _t: f64, y: &f64) -> ControlFlag {
        if *y > 0.9 * self.m {
            ControlFlag::Terminate("Reached 90% of carrying capacity".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}

fn main() {
    let mut method = DOP853::new().rtol(1e-12).atol(1e-12);
    let y0 = 1.0;
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_problem = ODEProblem::new(ode, t0, tf, y0);
    match logistic_growth_problem
        .even(2.0)  // sets t-out at interval dt: 2.0
        .solve(&mut method) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let Status::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("NumericalMethod stopped: {}", reason);
            }

            // Print the solution
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y);
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {}", solution.steps);
            println!("Rejected Steps: {}", solution.rejected_steps);
            println!("Accepted Steps: {}", solution.accepted_steps);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
