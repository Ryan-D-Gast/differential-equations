use differential_equations::ode::*;
use nalgebra::{Vector3, Vector6, vector};

/// Circular Restricted Three Body Problem (CR3BP)
pub struct Cr3bp {
    pub mu: f64, // CR3BP mass ratio
}

impl ODE<f64, 6> for Cr3bp {
    /// Differential equation for the initial value Circular Restricted Three
    /// Body Problem (CR3BP).
    /// All parameters are in non-dimensional form.
    fn diff(&self, _t: f64, y: &Vector6<f64>, dydt: &mut Vector6<f64>) {
        // Mass ratio
        let mu = self.mu;

        // Extracting states
        let (rx, ry, rz, vx, vy, vz) = (y[0], y[1], y[2], y[3], y[4], y[5]);

        // Distance to primary body
        let r13 = ((rx + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();
        // Distance to secondary body
        let r23 = ((rx - 1.0 + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();

        // Computing three-body dynamics
        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = vz;
        dydt[3] = rx + 2.0 * vy
            - (1.0 - mu) * (rx + mu) / r13.powi(3)
            - mu * (rx - 1.0 + mu) / r23.powi(3);
        dydt[4] = ry - 2.0 * vx - (1.0 - mu) * ry / r13.powi(3) - mu * ry / r23.powi(3);
        dydt[5] = -(1.0 - mu) * rz / r13.powi(3) - mu * rz / r23.powi(3);
    }
}

fn main() {
    // Solver with relative and absolute tolerances
    let mut solver = DOP853::new().rtol(1e-12).atol(1e-12); // DOP853 is one of the most accurate and efficient solvers and highly favored for Orbital Mechanics

    // Initialialize the CR3BP ode
    let ode = Cr3bp {
        mu: 0.012150585609624,
    }; // Earth-Moon ode

    // Initial conditions
    let sv = vector![
        // 9:2 L2 Southern NRHO orbit
        1.021881345465263,
        0.0,
        -0.182000000000000, // Position
        0.0,
        -0.102950816739606,
        0.0 // Velocity
    ];
    let t0 = 0.0;
    let tf = 3.0 * 1.509263667286943; // Period of the orbit (sv(t0) ~= sv(tf / 3.0))

    let cr3bp_ivp = IVP::new(ode, t0, tf, sv);

    fn extractor(y: &Vector6<f64>) -> Vector3<f64> {
        vector![y[3], y[4], y[5]]
    }

    // Solve the ode with even output at interval dt: 1.0
    match cr3bp_ivp
        .hyperplane_crossing(vector![1.0, 0.0, 0.0], vector![0.5, 0.5, 0.0], extractor, CrossingDirection::Both)
        .solve(&mut solver) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Print the solution
            println!("Solution:");
            println!("t, [x, y, z]");
            for (t, y) in solution.iter() {
                println!(
                    "{:.4}, [{:.4}, {:.4}, {:.4}]",
                    t, y[0], y[1], y[2]
                );
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
