//! Example 10: Custom Solution Output (Solout) for Pendulum
//! 
//! This example demonstrates a pendulum simulation with a custom Solout implementation
//! that captures specific events and calculates derived quantities. The pendulum equations are:
//! 
//! dθ/dt = ω
//! dω/dt = -(g/l)*sin(θ)
//! 
//! where:
//! - θ (theta) is the angle from vertical
//! - ω (omega) is the angular velocity
//! - g is the acceleration due to gravity
//! - l is the pendulum length
//!
//! The custom Solout tracks zero crossings (when the pendulum passes through vertical),
//! calculates energy at each point, and ensures points are appropriately spaced in time.
//!
//! This example illustrates:
//! - Creating and using a custom Solout implementation
//! - Tracking derived quantities during integration
//! - Event detection during solution output
//! - Energy conservation monitoring with high-accuracy solvers

use differential_equations::Solout;
use differential_equations::ode::*;
use nalgebra::{SVector, vector};
use std::f64::consts::PI;

/// Pendulum ode
struct Pendulum {
    g: f64, // Gravitational constant
    l: f64, // Length of pendulum
}

impl ODE<f64, SVector<f64, 2>> for Pendulum {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        // y[0] = theta (angle), y[1] = omega (angular velocity)
        dydt[0] = y[1];
        dydt[1] = -self.g / self.l * y[0].sin();
    }
}

/// Custom solout that:
/// 1. Captures points when angle crosses zero (pendulum passes vertical)
/// 2. Records energy at each point
/// 3. Ensures minimum spacing between points
struct PendulumSolout {
    g: f64,                   // Gravitational constant
    l: f64,                   // Length of pendulum
    last_angle: f64,          // Last angle to detect zero crossings
    min_dt: f64,              // Minimum time between output points
    last_output_time: f64,    // Last time a point was output
    energy_values: Vec<f64>,  // Store energy at each output point
    oscillation_count: usize, // Count of oscillations
}

impl PendulumSolout {
    fn new(g: f64, l: f64, min_dt: f64) -> Self {
        Self {
            g,
            l,
            last_angle: 0.0,
            min_dt,
            last_output_time: -f64::INFINITY,
            energy_values: Vec::new(),
            oscillation_count: 0,
        }
    }

    // Calculate the total energy of the pendulum
    fn calculate_energy(&self, y: &SVector<f64, 2>) -> f64 {
        let theta = y[0];
        let omega = y[1];

        // Kinetic energy: 0.5 * m * (l * omega)^2
        // Potential energy: m * g * l * (1 - cos(theta))
        // For simplicity, assume m=1
        let kinetic = 0.5 * self.l.powi(2) * omega.powi(2);
        let potential = self.g * self.l * (1.0 - theta.cos());

        kinetic + potential
    }
}

impl Solout<f64, SVector<f64, 2>> for PendulumSolout {
    fn solout<I>(
        &mut self,
        t_curr: f64,
        _t_prev: f64,
        y_curr: &SVector<f64, 2>,
        _y_prev: &SVector<f64, 2>,
        _interpolator: &mut I,
        solution: &mut Solution<f64, SVector<f64, 2>, String>,
    ) -> ControlFlag<String>
    where
        I: methods::adams::Interpolation<f64, SVector<f64, 2>>,
    {
        let current_angle = y_curr[0];
        let dt = t_curr - self.last_output_time;

        // Detect zero crossings (oscillation count)
        if self.last_angle.signum() != current_angle.signum() && current_angle.signum() != 0.0 {
            self.oscillation_count += 1;
        }

        // Add a point if:
        // 1. It's been at least min_dt since last point, AND
        // 2. Either:
        //    a. The angle crossed zero, OR
        //    b. The angle changed significantly
        let significant_change = (current_angle - self.last_angle).abs() > 0.2;
        let angle_crossed_zero = self.last_angle.signum() != current_angle.signum();

        if dt >= self.min_dt && (angle_crossed_zero || significant_change) {
            solution.push(t_curr, *y_curr);

            // Calculate and store energy
            self.energy_values.push(self.calculate_energy(y_curr));

            // Update tracking variables
            self.last_output_time = t_curr;
        }

        self.last_angle = current_angle;

        // Continue the integration
        ControlFlag::Continue
    }
}

fn main() {
    // Create pendulum ode
    let g = 9.81;
    let l = 1.0;
    let pendulum = Pendulum { g, l };

    // Initial conditions: angle = 30 degrees, angular velocity = 0
    let theta0 = 30.0 * PI / 180.0; // convert to radians
    let y0 = vector![theta0, 0.0];

    // Integration parameters
    let t0 = 0.0;
    let tf = 10.0;

    // Create custom solout
    let mut solout = PendulumSolout::new(g, l, 0.1);

    // Create solver and solve the IVP
    // Note DOP853 is so accurate the energy will remain almost constant. Other solvers will show some energy change due to lower accuracy.
    // This is why DOP853 is used a majority of the time for high accuracy simulations.
    let mut solver = DOP853::new().rtol(1e-8).atol(1e-8);
    let ivp = IVP::new(pendulum, t0, tf, y0);

    let result = ivp.solout(&mut solout).solve(&mut solver).unwrap();

    // Display results
    println!("Pendulum simulation results:");
    println!("Number of output points: {}", result.t.len());
    println!("Number of oscillations: {}", solout.oscillation_count / 2); // Divide by 2 because we count both crossings

    println!("\n Time   | Angle (rad) | Angular Vel | Energy");
    println!("------------------------------------------------");
    for (i, (t, y)) in result.iter().enumerate() {
        let energy = if i < solout.energy_values.len() {
            solout.energy_values[i]
        } else {
            0.0 // For t0/tf points that might not have energy calculated
        };

        println!(
            "{:6.3}  | {:11.6} | {:11.6} | {:11.6}",
            t, y[0], y[1], energy
        );
    }
}
