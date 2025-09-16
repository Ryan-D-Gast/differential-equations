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

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use std::f64::consts::PI;

struct Pendulum {
    g: f64,
    l: f64,
}

impl ODE<f64, SVector<f64, 2>> for Pendulum {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.g / self.l * y[0].sin();
    }
}

/// Custom solout that:
/// 1. Captures points when angle crosses zero (pendulum passes vertical)
/// 2. Records energy at each point
/// 3. Ensures minimum spacing between points
/// 4. Applies a boost to angular velocity when crossing zero
struct PendulumSolout {
    g: f64,                   // Gravitational constant
    l: f64,                   // Length of pendulum
    last_angle: f64,          // Last angle to detect zero crossings
    min_dt: f64,              // Minimum time between output points
    last_output_time: f64,    // Last time a point was output
    energy_values: Vec<f64>,  // Store energy at each output point
    oscillation_count: usize, // Count of oscillations
    boost_amount: f64,        // Angular velocity boost when crossing zero
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
            boost_amount: 0.05, // Default boost amount (5% increase in velocity)
        }
    }

    // Sets the boost amount
    fn with_boost(mut self, boost: f64) -> Self {
        self.boost_amount = boost;
        self
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
    ) -> ControlFlag<f64, SVector<f64, 2>, String>
    where
        I: Interpolation<f64, SVector<f64, 2>>,
    {
        let current_angle = y_curr[0];
        let dt = t_curr - self.last_output_time;

        // Detect zero crossings
        let angle_crossed_zero =
            self.last_angle.signum() != current_angle.signum() && current_angle.signum() != 0.0;

        if angle_crossed_zero {
            self.oscillation_count += 1;

            // Apply a boost to the angular velocity when crossing zero
            // Create a new state vector with the boosted velocity
            let mut boosted_state = *y_curr;

            // Add a boost in the direction the pendulum is moving
            // This preserves the direction of motion while increasing speed
            boosted_state[1] = y_curr[1] * (1.0 + self.boost_amount);

            // Calculate and store energy before adding the boost
            self.energy_values.push(self.calculate_energy(y_curr));

            // Add current state to solution before applying the boost
            solution.push(t_curr, *y_curr);

            // Update tracking variables
            self.last_output_time = t_curr;
            self.last_angle = current_angle;

            // Return ModifyState to update the solver with our boosted state
            return ControlFlag::ModifyState(t_curr, boosted_state);
        }

        // For non-zero crossing points, use the original logic
        let significant_change = (current_angle - self.last_angle).abs() > 0.2;

        if dt >= self.min_dt && significant_change {
            solution.push(t_curr, *y_curr);
            self.energy_values.push(self.calculate_energy(y_curr));

            self.last_output_time = t_curr;
        }

        // Continue the integration
        ControlFlag::Continue
    }
}

fn main() {
    // --- Problem Configuration ---
    let g = 9.81;
    let l = 1.0;
    let pendulum = Pendulum { g, l };
    let theta0 = 30.0 * PI / 180.0; // convert to radians
    let y0 = vector![theta0, 0.0];
    let t0 = 0.0;
    let tf = 10.0;
    let problem = ODEProblem::new(&pendulum, t0, tf, y0);

    // --- Solout and Solver Configuration ---
    let mut solout = PendulumSolout::new(g, l, 0.1).with_boost(0.05);
    let mut solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8);

    // --- Solve the ODE with custom solout ---
    let result = problem
        // The custom solout is applied to the problem here
        .solout(&mut solout)
        .solve(&mut solver)
        .unwrap();

    // Print the results
    println!("Pendulum simulation results:");
    println!("Number of output points: {}", result.t.len());
    println!("Number of oscillations: {}", solout.oscillation_count / 2); // Divide by 2 because we count both crossings
    println!("\n Time   | Angle (rad) | Angular Vel | Energy");
    println!("------------------------------------------------");
    for (i, (t, y)) in result.iter().enumerate() {
        let energy = solout.energy_values[i];

        println!(
            "{:6.3}  | {:11.6} | {:11.6} | {:11.6}",
            t, y[0], y[1], energy
        );
    }
}
