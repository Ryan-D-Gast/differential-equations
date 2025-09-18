//! Suite of test cases for checking the interpolation of DDE solvers.

use super::systems::ExponentialGrowth;
use differential_equations::{dde::DDEProblem, methods::ExplicitRungeKutta};
use nalgebra::vector;

macro_rules! test_dde_interpolation {
    (
        tolerance: $tolerance:expr,
        $(
            solver_name: $solver_name:ident, solver: $solver:expr
        ),+
    ) => {
        // Set initial conditions and system parameters
        let t0 = 0.0;
        let tf = 2.0;
        let y0 = vector![1.0];
        let k = 1.0;

        let phi = |_t: f64| {
            y0
        };

        $(
            // Define the system
            let system = ExponentialGrowth { k };

            // Create Initial Value Problem (DDEProblem) for the system
            // For L=0, the history function's exact form for t < t0 is less critical
            // as long as it provides a value at t0 if needed by an internal mechanism.
            // The primary check is for t > t0 via interpolation.
            let problem = DDEProblem::new(&system, t0, tf, y0, phi);

            // Initialize the solver
            let mut solver = $solver;

            // Points for interpolation
            let t_eval_points = vec![0.5, 1.0, 1.69];

            // Solve the system, requesting solutions at t_eval_points
            let results = problem
                .t_eval(t_eval_points.clone())
                .solve(&mut solver)
                .unwrap_or_else(|e| {
                    panic!("{} failed to solve: {:?}", stringify!($solver_name), e);
                });

            // Calculate the expected values using the exact solution y(t) = y0 * exp(k*t)
            for (idx, t_val) in t_eval_points.iter().enumerate() {
                let expected_y_val = y0[0] * (k * t_val).exp();
                let interpolated_y_val = results.y[idx][0];

                assert!(
                    (interpolated_y_val - expected_y_val).abs() < $tolerance,
                    "Interpolation failed for {} at t={}: Expected: {:.6e}, Got: {:.6e}, Diff: {:.3e}",
                    stringify!($solver_name),
                    t_val,
                    expected_y_val,
                    interpolated_y_val,
                    (interpolated_y_val - expected_y_val).abs()
                );
            }
            println!("Interpolation for {} passed.", stringify!($solver_name));
        )+
    };
}

#[test]
fn interpolation() {
    test_dde_interpolation! {
        tolerance: 1e-3,

        solver_name: RKV4, solver: ExplicitRungeKutta::rk4(0.1),
        solver_name: RKV655, solver: ExplicitRungeKutta::rkv655e(),
        solver_name: DOPRI5, solver: ExplicitRungeKutta::dopri5(),
        solver_name: DOP853, solver: ExplicitRungeKutta::dop853()
    }
}
