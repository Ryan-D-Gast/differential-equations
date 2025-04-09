use differential_equations::ode::*;
use nalgebra::{SVector, vector};

struct LogisticGrowth {
    k: f64,
    m: f64,
}

impl ODE<f64, 1, 1> for LogisticGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0] * (1.0 - y[0] / self.m);
    }

    fn event(&self, _t: f64, y: &SVector<f64, 1>) -> ControlFlag {
        if y[0] > 0.9 * self.m {
            ControlFlag::Terminate("Reached 90% of carrying capacity".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}

fn main() {
    let mut solver = DOP853::new().rtol(1e-12).atol(1e-12);
    let y0 = vector![1.0];
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_ivp = IVP::new(ode, t0, tf, y0);
    match logistic_growth_ivp
        .even(2.0)  // sets t-out at interval dt: 2.0
        .solve(&mut solver) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let SolverStatus::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("Solver stopped: {}", reason);
            }

            // Print the solution
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y[0]);
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
