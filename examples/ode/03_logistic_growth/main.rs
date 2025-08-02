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
use quill::*;

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
    let mut method = ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12);
    let y0 = 1.0;
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_problem = ODEProblem::new(ode, t0, tf, y0);
    match logistic_growth_problem
        .even(0.1)  // sets t-out at interval dt: 2.0
        .solve(&mut method) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let Status::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("Numerical Method stopped: {}", reason);
            }

            // Print the solution
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y);
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);

            // Plot the solution using quill
            Plot::builder()
                .title("Logistic Growth Model")
                .x_label("Time (t)")
                .y_label("Population")
                .y_range(Range::Manual { min: 0.0, max: 10.0 })
                .legend(Legend::BottomRightInside)
                .data([
                    Series::builder()
                        .name("Numerical Solution")
                        .color("Blue")
                        .data(
                            solution
                                .iter()
                                .map(|(t, y)| (*t, *y))
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                    Series::builder()
                        .name("Termination Asymptote")
                        .color("Red")
                        .data(
                            solution
                                .t
                                .iter()
                                .map(|t| (*t, 9.0)) // m = 10.0
                                .collect::<Vec<_>>(),
                        )
                        .line(Line::Dashed)
                        .build(),
                ])
                .build()
                .to_svg("examples/ode/03_logistic_growth/logistic_growth.svg")
                .expect("Failed to save plot as SVG");
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
