//! Suite of test cases for NumericalMethods vs results of SciPy using DOP853 & Tolerences = 1e-12

use super::systems::{ExponentialGrowth, HarmonicOscillator, LinearEquation, LogisticEquation};
use differential_equations::ode::ODEProblem;
use differential_equations::ode::methods::{
    adams::{APCF4, APCV4},
    runge_kutta::{
        explicit::{DOP853, DOPRI5, Euler, RK4, RKF, RKV65, RKV87, RKV98},
        implicit::{CrankNicolson, GaussLegendre4, GaussLegendre6, Radau5},
    },
};
use nalgebra::vector;

macro_rules! test_ode {
    (
        system_name: $system_name:ident,
        ode: $system:expr,
        t0: $t0:expr,
        tf: $tf:expr,
        y0: $y0:expr,
        expected_result: $expected_result:expr,
        $(solver_name: $solver_name:ident, solver: $solver:expr, tolerance: $tolerance:expr),+
    ) => {
        $(
            // Initialize the system
            let system = $system;

            // Set initial conditions
            let t0 = $t0;
            let tf = $tf;
            let y0 = $y0;

            // Create Initial Value Problem (ODEProblem) for the system
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Initialize the solver
            let mut solver = $solver;

            // Solve the system
            let results = problem.solve(&mut solver).unwrap();

            // Save results to csv
            results.to_csv(&format!("target/tests/ode/results/{}_{}.csv", stringify!($solver_name), stringify!($system_name))).unwrap();

            // Check the result against the expected result within the given tolerance
            let yf = results.y.last().unwrap();
            for i in 0..yf.len() {
                assert!(
                    (yf[i] - $expected_result[i]).abs() < $tolerance,
                    "{} {} failed: Expected: {:?}, Got: {:?}",
                    stringify!($solver_name),
                    stringify!($system_name),
                    $expected_result[i],
                    yf[i]
                );
            }
            println!("{} {} passed", stringify!($solver_name), stringify!($system_name));
        )+
    };
}

#[test]
fn accuracy() {
    test_ode! {
        system_name: exponential_growth_positive_direction,
        ode: ExponentialGrowth { k: 1.0 },
        t0: 0.0,
        tf: 10.0,
        y0: vector![1.0],
        expected_result: vector![22026.46579479],

        solver_name: DOP853,
        solver: DOP853::new().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e1,

        solver_name: RKF,
        solver: RKF::new(),
        tolerance: 1e3,

        solver_name: RK4,
        solver: RK4::new(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: Euler::new(0.01),
        tolerance: 2e3,

        solver_name: APCF4,
        solver: APCF4::new(0.01),
        tolerance: 1e3,

        solver_name: APCV4,
        solver: APCV4::new(),
        tolerance: 1e3,

        solver_name: RKV65,
        solver: RKV65::new(),
        tolerance: 1e-1,

        solver_name: RKV87,
        solver: RKV87::new(),
        tolerance: 1e-1,

        solver_name: RKV98,
        solver: RKV98::new(),
        tolerance: 1e-1,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: CrankNicolson::new(0.01),
        tolerance: 1e1,

        solver_name: GaussLegendre4,
        solver: GaussLegendre4::new(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: GaussLegendre6::new(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: Radau5::new(),
        tolerance: 1e-3
    }

    test_ode! {
        system_name: exponential_growth_negative_direction,
        ode: ExponentialGrowth { k: 1.0 },
        t0: 0.0,
        tf: -10.0,
        y0: vector![22026.46579479],
        expected_result: vector![1.0],

        // Explicit methods

        solver_name: DOP853,
        solver: DOP853::new().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e-2,

        solver_name: RKF,
        solver: RKF::new(),
        tolerance: 1e-2,

        solver_name: RK4,
        solver: RK4::new(-0.01),
        tolerance: 1e-2,

        solver_name: Euler,
        solver: Euler::new(-0.01),
        tolerance: 1e-1,

        solver_name: APCF4,
        solver: APCF4::new(-0.01),
        tolerance: 1e-1,

        solver_name: APCV4,
        solver: APCV4::new(),
        tolerance: 1e-1,

        solver_name: RKV65,
        solver: RKV65::new(),
        tolerance: 1e-2,

        solver_name: RKV87,
        solver: RKV87::new(),
        tolerance: 1e-2,

        solver_name: RKV98,
        solver: RKV98::new(),
        tolerance: 1e-2,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: CrankNicolson::new(-0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: GaussLegendre4::new(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: GaussLegendre6::new(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: Radau5::new(),
        tolerance: 1e-3
    }

    test_ode! {
        system_name: linear_equation,
        ode: LinearEquation { a: 1.0, b: 1.0 },
        t0: 0.0,
        tf: 10.0,
        y0: vector![1.0],
        expected_result: vector![44051.93158958],

        // Explicit methods

        solver_name: DOP853,
        solver: DOP853::new().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e1,

        solver_name: RKF,
        solver: RKF::new(),
        tolerance: 1e3,

        solver_name: RK4,
        solver: RK4::new(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: Euler::new(0.01),
        tolerance: 1e4,

        solver_name: APCF4,
        solver: APCF4::new(0.01),
        tolerance: 1e4,

        solver_name: APCV4,
        solver: APCV4::new(),
        tolerance: 1e4,

        solver_name: RKV65,
        solver: RKV65::new(),
        tolerance: 1e1,

        solver_name: RKV87,
        solver: RKV87::new(),
        tolerance: 1e1,

        solver_name: RKV98,
        solver: RKV98::new(),
        tolerance: 1e1,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: CrankNicolson::new(0.01),
        tolerance: 1e3,

        solver_name: GaussLegendre4,
        solver: GaussLegendre4::new(),
        tolerance: 1e1,

        solver_name: GaussLegendre6,
        solver: GaussLegendre6::new(),
        tolerance: 1e1,

        solver_name: Radau5,
        solver: Radau5::new(),
        tolerance: 1e1
    }

    test_ode! {
        system_name: harmonic_oscillator,
        ode: HarmonicOscillator { k: 1.0 },
        t0: 0.0,
        tf: 10.0,
        y0: vector![1.0, 0.0],
        expected_result: vector![-0.83907153, 0.54402111],

        // Explicit methods

        solver_name: DOP853,
        solver: DOP853::new().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e-2,

        solver_name: RKF,
        solver: RKF::new(),
        tolerance: 1e-3,

        solver_name: RK4,
        solver: RK4::new(0.01),
        tolerance: 1e-1,

        solver_name: Euler,
        solver: Euler::new(0.01),
        tolerance: 1e-1,

        solver_name: APCF4,
        solver: APCF4::new(0.01),
        tolerance: 1e-1,

        solver_name: APCV4,
        solver: APCV4::new(),
        tolerance: 1e-1,

        solver_name: RKV65,
        solver: RKV65::new(),
        tolerance: 1e-3,

        solver_name: RKV87,
        solver: RKV87::new(),
        tolerance: 1e-3,

        solver_name: RKV98,
        solver: RKV98::new(),
        tolerance: 1e-3,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: CrankNicolson::new(0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: GaussLegendre4::new(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: GaussLegendre6::new(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: Radau5::new(),
        tolerance: 1e-3
    }

    test_ode! {
        system_name: logistic_equation,
        ode: LogisticEquation { k: 1.0, m: 10.0 },
        t0: 0.0,
        tf: 10.0,
        y0: vector![0.1],
        expected_result: vector![9.95525518],

        // Explicit methods

        solver_name: DOP853,
        solver: DOP853::new().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: DOPRI5::new(),
        tolerance: 1e-3,

        solver_name: RKF,
        solver: RKF::new(),
        tolerance: 1e-2,

        solver_name: RK4,
        solver: RK4::new(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: Euler::new(0.01),
        tolerance: 1e-2,

        solver_name: APCF4,
        solver: APCF4::new(0.01),
        tolerance: 1e-2,

        solver_name: APCV4,
        solver: APCV4::new(),
        tolerance: 1e-2,

        solver_name: RKV65,
        solver: RKV65::new(),
        tolerance: 1e-3,

        solver_name: RKV87,
        solver: RKV87::new(),
        tolerance: 1e-3,

        solver_name: RKV98,
        solver: RKV98::new(),
        tolerance: 1e-3,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: CrankNicolson::new(0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: GaussLegendre4::new(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: GaussLegendre6::new(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: Radau5::new(),
        tolerance: 1e-3
    }
}
