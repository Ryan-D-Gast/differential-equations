//! Example 01: Exponential Growth
//!
//! This example demonstrates the solution of a simple exponential growth ODE:
//! dy/dt = k*y
//!
//! where:
//! - y is the quantity that grows exponentially
//! - k is the growth rate constant
//! - t is time
//!
//! The analytical solution to this ODE is y(t) = y₀*e^(k*t) where y₀ is the initial value.
//! Exponential growth models are used in various applications including population dynamics,
//! financial growth, and chemical reactions.
//!
//! This example showcases:
//! - Basic ODE definition and solution
//! - Setting custom tolerances for high accuracy
//! - Accessing solution statistics like step counts and evaluations

use differential_equations::prelude::*;
use quill::prelude::*;

// Differential equations are defined using structs that implement the ODE trait
struct ExponentialGrowth {
    k: f64,
}

impl ODE for ExponentialGrowth {
    // Define the differential equation dy/dt = k*y
    fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.k * y;
    }
}

fn main() {
    // Numerical methods have default settings, which can be customized via chaining methods
    // Here we use the DOP853 method, which is a high-order Runge-Kutta method
    let mut method = ExplicitRungeKutta::dop853()
        // Set the relative and absolute tolerances
        .rtol(1e-12)
        .atol(1e-12);

    // Define the ODE problem with initial conditions, this is known as a "initial value problem"
    let y0 = 1.0;
    let t0 = 0.0;
    let tf = 10.0;
    let ode = ExponentialGrowth { k: 1.0 };
    let exponential_growth_problem = ODEProblem::new(ode, t0, tf, y0);

    // ODE Problems are solved by chaining the solve method with the numerical method
    let solution = match exponential_growth_problem.solve(&mut method) {
        Ok(solution) => {
            // The solution struct contains the output (t,y) points, statistics, and other metadata
            solution
        }
        Err(e) => panic!("Error: {:?}", e),
    };

    // Print the solution using the fields of the Solution struct
    println!(
        "Solution: ({:?}, {:?})",
        solution.t.last().unwrap(),
        solution.y.last().unwrap()
    );
    println!("Function evaluations: {}", solution.evals.function);
    println!("Steps: {}", solution.steps.total());
    println!("Rejected Steps: {}", solution.steps.rejected);
    println!("Accepted Steps: {}", solution.steps.accepted);
    println!("Status: {:?}", solution.status);

    // Plot the solution using the quill library
    Plot::builder()
        .title("Exponential Growth ODE Solution")
        .x_label("Time (t)")
        .y_label("y(t)")
        .legend(Legend::TopLeftInside)
        .data([
            Series::builder()
                .name("Numerical Solution")
                .color("Blue")
                .data(solution.iter().map(|(t, y)| (*t, *y)).collect::<Vec<_>>())
                .marker(Marker::Circle)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Analytical Solution")
                .color("Red")
                .data({
                    let k = 1.0;
                    let y0 = 1.0;
                    (0..=100)
                        .map(|i| {
                            let t = i as f64 * 0.1;
                            let analytical = y0 * (k * t).exp();
                            (t, analytical)
                        })
                        .collect::<Vec<_>>()
                })
                .marker(Marker::Circle)
                .marker_size(3.0)
                .line(Line::None)
                .build(),
        ])
        .build()
        .to_svg("examples/ode/01_exponential_growth/exponential_growth.svg")
        .expect("Failed to save plot as SVG");
}
