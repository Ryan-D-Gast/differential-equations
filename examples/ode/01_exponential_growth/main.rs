use differential_equations::ode::*;
use nalgebra::{SVector, vector};

// Define the ode
struct ExponentialGrowth {
    k: f64,
}

// Implement the ODE trait for the ExponentialGrowth ode
// Notice instead of ODE<f64, 1, 1> which matches the defaults for the generic parameters, we can just use ODE
impl ODE for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0];
    }
}

fn main() {
    // Initialize the solver
    let mut solver = DOP853::new() // Normal refers to default with t-out being all t-steps e.g. no interpolation
        .rtol(1e-12) // Set the relative tolerance, Default is 1e-3 for DOP853
        .atol(1e-12); // Set the absolute tolerance, Default is 1e-6 for DOP853

    // Initialize the initial value problem
    let y0 = vector![1.0]; // vector! is a nalgebra macro to create a SVector; functions similarly to vec! but creates a static vector e.g. not dynamic and has a fixed size
    let t0 = 0.0;
    let tf = 10.0;
    let ode = ExponentialGrowth { k: 1.0 };
    let exponential_growth_ivp = IVP::new(ode, t0, tf, y0);

    // Solve the initial value problem
    let solution = match exponential_growth_ivp.solve(&mut solver) {
        Ok(solution) => solution,
        Err(e) => panic!("Error: {:?}", e),
    };

    // Print the solution using the fields of the Solution struct, which is returned by the solve method
    println!(
        "Solution: ({:?}, {:?})",
        solution.t.last().unwrap(),
        solution.y.last().unwrap()
    );
    println!("Function evaluations: {}", solution.evals);
    println!("Steps: {}", solution.steps);
    println!("Rejected Steps: {}", solution.rejected_steps);
    println!("Accepted Steps: {}", solution.accepted_steps);
    println!("Status: {:?}", solution.status);
}
