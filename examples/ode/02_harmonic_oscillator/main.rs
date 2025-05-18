//! Example 02: Harmonic Oscillator
//!
//! This example simulates a simple harmonic oscillator using the system:
//! dx/dt = v
//! dv/dt = -k*x
//!
//! where:
//! - x is the position
//! - v is the velocity
//! - k is the spring constant (normalized for unit mass)
//! - t is time
//!
//! The harmonic oscillator is a fundamental model in physics, representing systems
//! that experience a restoring force proportional to displacement, such as springs,
//! pendulums (for small angles), and simple electronic circuits.
//!
//! This example demonstrates:
//! - Solving a system of first-order ODEs (written as a vector ODE)
//! - Using the nalgebra library for vector state representation
//! - Compact solution approach with minimal code

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use plotlars::{Axis, LinePlot, TickDirection, Plot};

struct HarmonicOscillator {
    k: f32,
}

impl ODE<f32, SVector<f32, 2>> for HarmonicOscillator {
    fn diff(&self, _t: f32, y: &SVector<f32, 2>, dydt: &mut SVector<f32, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.k * y[0];
    }
}

fn main() {
    // Note how unlike 01_exponential_growth/main.rs, no intermediate variables are used and the ODEProblem is setup and solved in one step.
    let solution =
        match ODEProblem::new(HarmonicOscillator { k: 1.0 }, 0.0, 10.0, vector![1.0, 0.0])
            .solve(&mut RK4::new(0.01))
        {
            Ok(solution) => solution,
            Err(e) => panic!("Error: {:?}", e),
        };
    let (tf, yf) = solution.last().unwrap();
    println!("Solution: ({:?}, {:?})", tf, yf);

    // Plotting the solution using Plotlars
    let df = solution.to_polars().unwrap();
    LinePlot::builder()
        .data(&df)
        .x("t")
        .y("y0")
        .plot_title("Harmonic Oscillator")
        .x_title("Time (s)")
        .y_title("Position (m)")
        .x_axis(
            &Axis::new()
                .tick_direction(TickDirection::InSide)
                
        )
        .y_axis(
            &Axis::new()
                .tick_direction(TickDirection::InSide)
        )  
        .build()
        .plot();
}
