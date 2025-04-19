use differential_equations::ode::methods::RKF;
use differential_equations::ode::*;
use nalgebra::{SVector, vector};

/// Damped Pendulum Model
///
/// This struct defines the parameters for the damped pendulum model.
struct DampedPendulumModel {
    g: f64, // Acceleration due to gravity (m/s^2)
    l: f64, // Length of the pendulum (m)
    b: f64, // Damping coefficient (kg/s)
    m: f64, // Mass of the pendulum bob (kg)
}

impl ODE<f64, SVector<f64, 2>> for DampedPendulumModel {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        let theta = y[0]; // Angle (radians)
        let omega = y[1]; // Angular velocity (radians/s)

        dydt[0] = omega; // dtheta/dt = omega
        dydt[1] = -(self.b / self.m) * omega - (self.g / self.l) * theta.sin(); // domega/dt = -(b/m)*omega - (g/l)*sin(theta)
    }

    fn event(&self, _t: f64, y: &SVector<f64, 2>) -> ControlFlag {
        let theta = y[0];
        let omega = y[1];

        // Terminate the simulation when the pendulum is close to equilibrium (theta and omega are close to 0)
        if theta.abs() < 0.01 && omega.abs() < 0.01 {
            ControlFlag::Terminate("Pendulum reached equilibrium".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}

fn main() {
    // Initialize method with relative and absolute tolerances
    let mut method = RKF::new();

    // Initial conditions
    let initial_angle = 1.0; // Initial angle (radians)
    let initial_velocity = 0.0; // Initial angular velocity (radians/s)

    let y0 = vector![initial_angle, initial_velocity];
    let t0 = 0.0;
    let tf = 100.0;

    // Pendulum parameters
    let g = 9.81; // Acceleration due to gravity (m/s^2)
    let l = 1.0; // Length of the pendulum (m)
    let b = 0.2; // Damping coefficient (kg/s)
    let m = 1.0; // Mass of the pendulum bob (kg)

    let ode = DampedPendulumModel { g, l, b, m };
    let pendulum_ivp = IVP::new(ode, t0, tf, y0);

    // t-out points
    let t_out = vec![0.0, 1.0, 3.0, 4.5, 6.9, 10.0];

    // Solve the ode with even output at interval dt: 0.1
    match pendulum_ivp.t_eval(t_out).solve(&mut method) {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let Status::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("NumericalMethod stopped: {:?}", reason);
            }

            // Print the solution
            println!("Solution:");
            println!("Time, Angle (radians), Angular Velocity (radians/s)");
            for (t, y) in solution.iter() {
                println!("{:.4}, {:.4}, {:.4}", t, y[0], y[1]);
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
