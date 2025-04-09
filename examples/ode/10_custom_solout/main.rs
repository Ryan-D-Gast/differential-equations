use differential_equations::ode::*;
use nalgebra::{SVector, vector};
use std::f64::consts::PI;

/// Pendulum ode
struct Pendulum {
    g: f64, // Gravitational constant
    l: f64, // Length of pendulum
}

impl ODE<f64, 2> for Pendulum {
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

impl Solout<f64, 2, 1> for PendulumSolout {
    fn solout<S>(&mut self, solver: &mut S, solution: &mut Solution<f64, 2, 1, String>)
    where
        S: Solver<f64, 2, 1, String> 
    {
        let t = solver.t();
        let y = solver.y();

        let current_angle = y[0];
        let dt = t - self.last_output_time;

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
            solution.push(t, *y);

            // Calculate and store energy
            self.energy_values.push(self.calculate_energy(y));

            // Update tracking variables
            self.last_output_time = t;
        }

        self.last_angle = current_angle;
    }

    // Do not include t0 and tf in output
    //
    // Note that for t0 solout is NOT called untill a single step has been completed so that t != t_prev and cause any NaN issues.
    // In addition because the Solout doesn't log every step it is almost certain that tf will not have a energy value calculated.
    // Because of this to make sure lenght of the energy values is same as the solution points we return false.
    // Of course this is optional and user can choose to include t0 and tf if they want. This demonstrates how to do it and why it might be desired.
    fn include_t0_tf(&self) -> bool {
        false
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
