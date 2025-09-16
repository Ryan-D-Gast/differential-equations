//! Example 05: Damped Pendulum
//!
//! This example simulates a damped pendulum using the system:
//! dθ/dt = ω
//! dω/dt = -(b/m)*ω - (g/l)*sin(θ)
//!
//! where:
//! - θ (theta) is the angle from vertical
//! - ω (omega) is the angular velocity
//! - b is the damping coefficient
//! - m is the mass of the pendulum bob
//! - g is the acceleration due to gravity
//! - l is the length of the pendulum
//!
//! Damped pendulums demonstrate both oscillatory behavior and decay due to friction,
//! eventually settling at the equilibrium position. This system appears in mechanical
//! engineering, clock design, and as a model of various control systems.
//!
//! This example demonstrates:
//! - Event detection to terminate simulation at equilibrium
//! - Evaluation at specific time points (t_eval)
//! - Status checking with simulation results

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

struct DampedPendulumModel {
    g: f64,
    l: f64,
    b: f64,
    m: f64,
}

impl ODE<f64, SVector<f64, 2>> for DampedPendulumModel {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        let theta = y[0];
        let omega = y[1];

        dydt[0] = omega;
        dydt[1] = -(self.b / self.m) * omega - (self.g / self.l) * theta.sin();
    }
}

impl Event<f64, SVector<f64, 2>> for DampedPendulumModel {
    fn config(&self) -> EventConfig {
        EventConfig::default().terminal() // Will terminate after the first event
    }

    /// Event function to detect when the pendulum is near equilibrium
    fn event(&self, _t: f64, y: &SVector<f64, 2>) -> f64 {
        let theta = y[0];
        let omega = y[1];
        // Event function g(t,y) = max(|theta|, |omega|) - 0.01
        theta.abs().max(omega.abs()) - 0.01
    }
}

fn main() {
    // --- Problem Configuration ---
    let initial_angle = 1.0; // Initial angle (radians)
    let initial_velocity = 0.0; // Initial angular velocity (radians/s)
    let y0 = vector![initial_angle, initial_velocity];
    let t0 = 0.0;
    let tf = 100.0;

    let g = 9.81; // Acceleration due to gravity (m/s^2)
    let l = 1.0; // Length of the pendulum (m)
    let b = 0.2; // Damping coefficient (kg/s)
    let m = 1.0; // Mass of the pendulum bob (kg)
    let ode = DampedPendulumModel { g, l, b, m };
    let pendulum_problem = ODEProblem::new(&ode, t0, tf, y0);

    // --- Numerically Solve the ODE ---
    let mut method = ExplicitRungeKutta::rkf45();
    let t_out = [0.0, 1.0, 3.0, 4.5, 6.9, 10.0];
    match pendulum_problem.t_eval(t_out).event(&ode).solve(&mut method) {
        Ok(solution) => {
            // Check if the solver stopped early due to the event condition
            if let Status::Interrupted = solution.status {
                println!("NumericalMethod stopped: Pendulum reached near equilibrium");
            }

            // Print the solution
            println!("Solution:");
            println!("Time, Angle (radians), Angular Velocity (radians/s)");
            for (t, y) in solution.iter() {
                println!("{:.4}, {:.4}, {:.4}", t, y[0], y[1]);
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
