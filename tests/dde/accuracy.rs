//! Suite of test cases for DDE NumericalMethods.
//! Expected results should be verified against a trusted solver.

use super::systems::MackeyGlass;
use differential_equations::dde::DDEProblem;
use differential_equations::dde::methods::{BS23, DOPRI5}; // DDE23 and DDE45 are aliases for these
use nalgebra::vector;
use std::fs;

macro_rules! test_dde {
    (
        system_name: $system_name:ident,
        dde: $system:expr,
        t0: $t0:expr,
        tf: $tf:expr,
        y0: $y0:expr,
        history: $history:expr,
        expected_result: $expected_result:expr,
        $(solver_name: $solver_name:ident, solver: $solver:expr, tolerance: $tolerance:expr),+
    ) => {
        $(
            // Initialize the system
            let system = $system;

            // Set initial conditions and time span
            let t0 = $t0;
            let tf = $tf;
            let y0 = $y0;
            let history_fn = $history;

            // Create Initial Value Problem (DDEProblem) for the system
            let problem = DDEProblem::new(system, t0, tf, y0, history_fn);

            // Initialize the solver
            let mut solver = $solver;

            // Create directory for results if it doesn't exist
            let results_dir = format!("target/tests/dde/results");
            fs::create_dir_all(&results_dir).expect("Failed to create DDE results directory");

            // Solve the system
            let results = problem.solve(&mut solver).unwrap_or_else(|e| {
                panic!("{} {} failed to solve: {:?}", stringify!($solver_name), stringify!($system_name), e);
            });

            // Save results to csv
            results.to_csv(&format!("{}/{}_{}.csv", results_dir, stringify!($solver_name), stringify!($system_name))).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to save CSV for {} {}: {:?}", stringify!($solver_name), stringify!($system_name), e);
            });

            // Check the result against the expected result within the given tolerance
            let yf = results.y.last().unwrap_or_else(|| {
                panic!("{} {} produced no results", stringify!($solver_name), stringify!($system_name));
            });

            for i in 0..yf.len() {
                assert!(
                    (yf[i] - $expected_result[i]).abs() < $tolerance,
                    "{} {} failed: Expected: {:?}, Got: {:?}. Difference: {:.6e}",
                    stringify!($solver_name),
                    stringify!($system_name),
                    $expected_result[i],
                    yf[i],
                    (yf[i] - $expected_result[i]).abs()
                );
            }
            println!("{} {} passed", stringify!($solver_name), stringify!($system_name));
        )+
    };
}

#[test]
fn accuracy() {
    // Define parameters for the Mackey-Glass system
    let mackey_glass_params = MackeyGlass {
        beta: 0.2,
        gamma: 0.1,
        n: 10.0,
        tau: 17.0,
    };

    // Define initial conditions and time span
    let t0 = 0.0;
    let tf = 50.0; // Time horizon for the simulation
    let y0 = vector![0.5]; // Initial state y(0)

    // Define the history function: y(t) = 0.5 for t <= 0
    let history_fn = |_t: f64| vector![0.5];

    // IMPORTANT: The expected_result is a placeholder.
    // Replace with a value obtained from a trusted DDE solver for these parameters.
    let placeholder_expected_result_at_tf50 = vector![0.6441197095478753];

    test_dde! {
        system_name: mackey_glass_default_params,
        dde: mackey_glass_params.clone(),
        t0: t0,
        tf: tf,
        y0: y0,
        history: history_fn,
        expected_result: placeholder_expected_result_at_tf50,

        solver_name: BS23,
        solver: BS23::new(),
        tolerance: 1e-2,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e-2
    }
}
