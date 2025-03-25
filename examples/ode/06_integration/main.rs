use differential_equations::ode::*;
use differential_equations::ode::solvers::RKF;
use nalgebra::{SVector, vector};

/// Define the system for integration.
/// In this example, we will integrate a simple function: y' = t.
#[derive(Clone)]
struct IntegrationSystem;

impl System for IntegrationSystem {
    fn diff(&self, t: f64, _y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = t;
    }
}

fn main() {
    // Define the initial value problem.
    let system = IntegrationSystem;
    let t0 = 0.0;
    let tf = 5.0;
    let y0 = vector![0.0];

    let ivp = IVP::new(system, t0, tf, y0);

    // Create a solver (RKF in this case).
    let mut solver = RKF::new(0.01);

    // Solve the IVP.
    let solution = ivp.solve(&mut solver).unwrap();

    // Print the results.
    println!("Numerical Integration Example:");
    println!("-----------------------------");
    println!("t\t\ty");

    for (t, y) in solution.iter() {
        println!("{:.6}\t{:.6}", t, y[0]);
    }

    // Verify the result. The analytical solution of y' = t with y(0) = 0 is y = t^2 / 2.
    let analytical_solution = tf.powi(2) / 2.0;
    let numerical_solution = solution.y.last().unwrap()[0];
    let error = (analytical_solution - numerical_solution).abs();

    println!("-----------------------------");
    println!("Analytical Solution at tf: {:.6}", analytical_solution);
    println!("Numerical Solution at tf: {:.6}", numerical_solution);
    println!("Absolute Error: {:.6}", error);

    // Example with dense output
    println!("-----------------------------");
    println!("Dense Output Example:");
    let ivp_dense = ivp.dense(2); // 5 interpolation points between each step
    let mut solver_dense = RKF::new(0.01);
    let solution_dense = ivp_dense.solve(&mut solver_dense).unwrap();

    println!("t\t\ty");
    for (t, y) in solution_dense.iter() {
        println!("{:.6}\t{:.6}", t, y[0]);
    }

    // Example with even t-out
    println!("-----------------------------");
    println!("Even t-out Example:");
    let ivp_even = ivp.even(1.0); // t-out at interval dt: 1.0
    let mut solver_even = RKF::new(0.01);
    let solution_even = ivp_even.solve(&mut solver_even).unwrap();

    println!("t\t\ty");
    for (t, y) in solution_even.iter() {
        println!("{:.6}\t{:.6}", t, y[0]);
    }

    // Example with t-out points
    println!("-----------------------------");
    println!("t-out Points Example:");
    let t_out = vec![0.0, 2.0, 5.0];
    let ivp_t_out = ivp.t_eval(t_out);
    let mut solver_t_out = RKF::new(0.01);
    let solution_t_out = ivp_t_out.solve(&mut solver_t_out).unwrap();

    println!("t\t\ty");
    for (t, y) in solution_t_out.iter() {
        println!("{:.6}\t{:.6}", t, y[0]);
    }
}