use differential_equations::ode::*;
use differential_equations::derive::State;
use nalgebra::{Vector3, vector};

/// Circular Restricted Three Body Problem (CR3BP)
pub struct Cr3bp {
    pub mu: f64, // CR3BP mass ratio
}

impl ODE<f64, StateVector<f64>> for Cr3bp {
    /// Differential equation for the initial value Circular Restricted Three
    /// Body Problem (CR3BP).
    /// All parameters are in non-dimensional form.
    fn diff(&self, _t: f64, sv: &StateVector<f64>, dsdt: &mut StateVector<f64>) {
        // Mass ratio
        let mu = self.mu;

        // Distance to primary body
        let r13 = ((sv.x + mu).powi(2) + sv.y.powi(2) + sv.z.powi(2)).sqrt();
        // Distance to secondary body
        let r23 = ((sv.x - 1.0 + mu).powi(2) + sv.y.powi(2) + sv.z.powi(2)).sqrt();

        // Computing three-body dynamics
        dsdt.x = sv.vx;
        dsdt.y = sv.vy;
        dsdt.z = sv.vz;
        dsdt.vx = sv.x + 2.0 * sv.vy
            - (1.0 - mu) * (sv.x + mu) / r13.powi(3)
            - mu * (sv.x - 1.0 + mu) / r23.powi(3);
        dsdt.vy = sv.y - 2.0 * sv.vx - (1.0 - mu) * sv.y / r13.powi(3) - mu * sv.y / r23.powi(3);
        dsdt.vz = -(1.0 - mu) * sv.z / r13.powi(3) - mu * sv.z / r23.powi(3);
    }
}

#[derive(State)]
pub struct StateVector<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub vx: T,
    pub vy: T,
    pub vz: T,
}

fn main() {
    // Initialize method with relative and absolute tolerances
    let mut method = DOP853::new().rtol(1e-12).atol(1e-12); // DOP853 is one of the most accurate and efficient solvers and highly favored for Orbital Mechanics

    // Initialialize the CR3BP ode
    let ode = Cr3bp {
        mu: 0.012150585609624,
    }; // Earth-Moon ode

    // Initial conditions
    let sv = StateVector {
        x: 1.021881345465263,
        y: 0.0,
        z: -0.182000000000000,
        vx: 0.0,
        vy: -0.102950816739606,
        vz: 0.0
    };
    let t0 = 0.0;
    let tf = 3.0 * 1.509263667286943; // Period of the orbit (sv(t0) ~= sv(tf / 3.0))

    let cr3bp_ivp = IVP::new(ode, t0, tf, sv);

    fn extractor(sv: &StateVector<f64>) -> Vector3<f64> {
        vector![sv.x, sv.y, sv.z]
    }

    // Solve the ode with even output at interval dt: 1.0
    match cr3bp_ivp
        .hyperplane_crossing(vector![1.0, 0.0, 0.0], vector![0.5, 0.5, 0.0], extractor, CrossingDirection::Both)
        .solve(&mut method) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Print the solution
            println!("Solution:");
            println!("t, [x, y, z]");
            for (t, sv) in solution.iter() {
                println!(
                    "{:.4}, [{:.4}, {:.4}, {:.4}]",
                    t, sv.x, sv.y, sv.z
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
