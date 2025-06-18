//! Suite of test cases for checking the interpolation of the solvers.

use super::systems;
use differential_equations::{
    methods::ExplicitRungeKutta,
    ode::{
        ODEProblem,
        methods::{
            adams::{APCF4, APCV4},
            runge_kutta::{
                implicit::{CrankNicolson, GaussLegendre6, Radau5},
            },
        }
    }
};
use nalgebra::vector;
use systems::ExponentialGrowth;

macro_rules! test_interpolation {
    (
        tolerance: $tolerance:expr,
        $(
            solver_name: $solver_name:ident, solver: $solver:expr
        ),+
    ) => {
        // Set initial conditions
        let t0 = 0.0;
        let tf = 2.0;
        let y0 = vector![1.0];

        $(
            // Define the system
            let system = ExponentialGrowth { k: 1.0 };

            // Create Initial Value Problem (ODEProblem) for the system
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Initialize the solver
            let mut solver = $solver;

            // Solve the system
            let results = problem
                .t_eval(
                    vec![0.5, 1.0, 1.69] // Get the Point at t = 0.5, 1.0, 1.69
                )
                .solve(&mut solver)
                .unwrap();

            // Calculate the expected value using the exact solution
            let expected_y = vector![f64::exp(0.5), f64::exp(1.0), f64::exp(1.69)];

            // Check if the interpolated value is close to the expected value
            assert!(
                (results.y[0][0] - expected_y[0]).abs() < $tolerance,
                "Interpolation failed for {}: Expected: {:?}, Got: {:?}",
                stringify!($solver_name),
                expected_y[0],
                results.y[0][0]
            );
        )+
    };
}

#[test]
fn interpolation() {
    test_interpolation! {
        tolerance: 1e-3,
        // This method uses a internal high order interpolation method
        solver_name: DOP853, solver: ExplicitRungeKutta::dop853(),
        solver_name: DOPRI5, solver: ExplicitRungeKutta::dopri5(),
        solver_name: RKV65, solver: ExplicitRungeKutta::rkv655e(),
        solver_name: RKV87, solver: ExplicitRungeKutta::rkv877e(),
        solver_name: RKV98, solver: ExplicitRungeKutta::rkv988e(),

        // These methods use cubic Hermite interpolation
        solver_name: RKF, solver: ExplicitRungeKutta::rkf45(),
        solver_name: RK4, solver: ExplicitRungeKutta::rk4(0.01),
        solver_name: APCF4, solver: APCF4::new(0.01),
        solver_name: APCV4, solver: APCV4::new().h0(0.01),
        solver_name: CrankNicolson, solver: CrankNicolson::new(0.01),
        solver_name: GaussLegendre6, solver: GaussLegendre6::new().h0(0.01),
        solver_name: Radau5, solver: Radau5::new().h0(0.01)
    }

    test_interpolation! {
        tolerance: 1e-2,
        // Euler's method produces less accurate y values thus affecting the interpolation
        // cubic Hermite interpolation is used.
        solver_name: Euler, solver: ExplicitRungeKutta::euler(0.01)
    }
}
