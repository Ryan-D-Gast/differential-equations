//! Suite of test cases for numerical methods vs results of SciPy using DOP853 & Tolerences = 1e-12

use super::systems::{
    ExponentialGrowth, HarmonicOscillator, HiresProblem, LinearEquation, LogisticEquation,
    RobertsonProblem,
};
use differential_equations::{
    methods::{
        AdamsPredictorCorrector, DiagonallyImplicitRungeKutta, ExplicitRungeKutta,
        ImplicitRungeKutta,
    },
    ode::ODEProblem,
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
        solver: ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5(),
        tolerance: 1e1,

        solver_name: RKF45,
        solver: ExplicitRungeKutta::rkf45(),
        tolerance: 1e3,

        solver_name: CashKarp,
        solver: ExplicitRungeKutta::cash_karp(),
        tolerance: 1e3,

        solver_name: RK4,
        solver: ExplicitRungeKutta::rk4(0.01),
        tolerance: 1e3,

        solver_name: ThreeEighths,
        solver: ExplicitRungeKutta::three_eighths(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: ExplicitRungeKutta::euler(0.01),
        tolerance: 2e3,

        solver_name: Midpoint,
        solver: ExplicitRungeKutta::midpoint(0.01),
        tolerance: 1e3,

        solver_name: Heun,
        solver: ExplicitRungeKutta::heun(0.01),
        tolerance: 1e3,

        solver_name: Ralston,
        solver: ExplicitRungeKutta::ralston(0.01),
        tolerance: 1e3,

        solver_name: APCF4,
        solver: AdamsPredictorCorrector::f4(0.01),
        tolerance: 1e3,

        solver_name: APCV4,
        solver: AdamsPredictorCorrector::v4(),
        tolerance: 1e3,

        solver_name: RKV65,
        solver: ExplicitRungeKutta::rkv655e(),
        tolerance: 1e-1,

        solver_name: RKV766e,
        solver: ExplicitRungeKutta::rkv766e(),
        tolerance: 1e-1,

        solver_name: RKV767e,
        solver: ExplicitRungeKutta::rkv767e(),
        tolerance: 1e-1,

        solver_name: RKV877e,
        solver: ExplicitRungeKutta::rkv877e(),
        tolerance: 1e-1,

        solver_name: RKV878e,
        solver: ExplicitRungeKutta::rkv878e(),
        tolerance: 1e-1,

        solver_name: RKV988e,
        solver: ExplicitRungeKutta::rkv988e(),
        tolerance: 1e-1,

        solver_name: RKV989e,
        solver: ExplicitRungeKutta::rkv989e(),
        tolerance: 1e-1,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(0.01),
        tolerance: 1e1,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
        tolerance: 1e1
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
        solver: ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5(),
        tolerance: 1e-2,

        solver_name: RKF45,
        solver: ExplicitRungeKutta::rkf45(),
        tolerance: 1e-2,

        solver_name: CashKarp,
        solver: ExplicitRungeKutta::cash_karp(),
        tolerance: 1e-2,

        solver_name: RK4,
        solver: ExplicitRungeKutta::rk4(-0.01),
        tolerance: 1e-2,

        solver_name: ThreeEighths,
        solver: ExplicitRungeKutta::three_eighths(-0.01),
        tolerance: 1e-2,

        solver_name: Euler,
        solver: ExplicitRungeKutta::euler(-0.01),
        tolerance: 1e-1,

        solver_name: Midpoint,
        solver: ExplicitRungeKutta::midpoint(-0.01),
        tolerance: 1e-1,

        solver_name: Heun,
        solver: ExplicitRungeKutta::heun(-0.01),
        tolerance: 1e-1,

        solver_name: Ralston,
        solver: ExplicitRungeKutta::ralston(-0.01),
        tolerance: 1e-1,

        solver_name: APCF4,
        solver: AdamsPredictorCorrector::f4(-0.01),
        tolerance: 1e-1,

        solver_name: APCV4,
        solver: AdamsPredictorCorrector::v4(),
        tolerance: 1e-1,

        solver_name: RKV65,
        solver: ExplicitRungeKutta::rkv655e(),
        tolerance: 1e-2,

        solver_name: RKV766e,
        solver: ExplicitRungeKutta::rkv766e(),
        tolerance: 1e-2,

        solver_name: RKV767e,
        solver: ExplicitRungeKutta::rkv767e(),
        tolerance: 1e-2,

        solver_name: RKV877e,
        solver: ExplicitRungeKutta::rkv877e(),
        tolerance: 1e-2,

        solver_name: RKV878e,
        solver: ExplicitRungeKutta::rkv878e(),
        tolerance: 1e-2,

        solver_name: RKV988e,
        solver: ExplicitRungeKutta::rkv988e(),
        tolerance: 1e-2,

        solver_name: RKV989e,
        solver: ExplicitRungeKutta::rkv989e(),
        tolerance: 1e-2,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(-0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
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
        solver: ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5(),
        tolerance: 1e1,

        solver_name: RKF45,
        solver: ExplicitRungeKutta::rkf45(),
        tolerance: 1e3,

        solver_name: CashKarp,
        solver: ExplicitRungeKutta::cash_karp(),
        tolerance: 1e3,

        solver_name: RK4,
        solver: ExplicitRungeKutta::rk4(0.01),
        tolerance: 1e3,

        solver_name: ThreeEighths,
        solver: ExplicitRungeKutta::three_eighths(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: ExplicitRungeKutta::euler(0.01),
        tolerance: 1e4,

        solver_name: Midpoint,
        solver: ExplicitRungeKutta::midpoint(0.01),
        tolerance: 1e4,

        solver_name: Heun,
        solver: ExplicitRungeKutta::heun(0.01),
        tolerance: 1e4,

        solver_name: Ralston,
        solver: ExplicitRungeKutta::ralston(0.01),
        tolerance: 1e4,

        solver_name: APCF4,
        solver: AdamsPredictorCorrector::f4(0.01),
        tolerance: 1e4,

        solver_name: APCV4,
        solver: AdamsPredictorCorrector::v4(),
        tolerance: 1e4,

        solver_name: RKV65,
        solver: ExplicitRungeKutta::rkv655e(),
        tolerance: 1e1,

        solver_name: RKV766e,
        solver: ExplicitRungeKutta::rkv766e(),
        tolerance: 1e1,

        solver_name: RKV767e,
        solver: ExplicitRungeKutta::rkv767e(),
        tolerance: 1e1,

        solver_name: RKV877e,
        solver: ExplicitRungeKutta::rkv877e(),
        tolerance: 1e1,

        solver_name: RKV878e,
        solver: ExplicitRungeKutta::rkv878e(),
        tolerance: 1e1,

        solver_name: RKV988e,
        solver: ExplicitRungeKutta::rkv988e(),
        tolerance: 1e1,

        solver_name: RKV989e,
        solver: ExplicitRungeKutta::rkv989e(),
        tolerance: 1e1,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(0.01),
        tolerance: 1e3,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4(),
        tolerance: 1e1,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6(),
        tolerance: 1e1,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
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
        solver: ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5(),
        tolerance: 1e-2,

        solver_name: RKF45,
        solver: ExplicitRungeKutta::rkf45(),
        tolerance: 1e-3,

        solver_name: CashKarp,
        solver: ExplicitRungeKutta::cash_karp(),
        tolerance: 1e-3,

        solver_name: RK4,
        solver: ExplicitRungeKutta::rk4(0.01),
        tolerance: 1e-1,

        solver_name: ThreeEighths,
        solver: ExplicitRungeKutta::three_eighths(0.01),
        tolerance: 1e-1,

        solver_name: Euler,
        solver: ExplicitRungeKutta::euler(0.01),
        tolerance: 1e-1,

        solver_name: Midpoint,
        solver: ExplicitRungeKutta::midpoint(0.01),
        tolerance: 1e-1,

        solver_name: Heun,
        solver: ExplicitRungeKutta::heun(0.01),
        tolerance: 1e-1,

        solver_name: Ralston,
        solver: ExplicitRungeKutta::ralston(0.01),
        tolerance: 1e-1,

        solver_name: APCF4,
        solver: AdamsPredictorCorrector::f4(0.01),
        tolerance: 1e-1,

        solver_name: APCV4,
        solver: AdamsPredictorCorrector::v4(),
        tolerance: 1e-1,

        solver_name: RKV65,
        solver: ExplicitRungeKutta::rkv655e(),
        tolerance: 1e-3,

        solver_name: RKV766e,
        solver: ExplicitRungeKutta::rkv766e(),
        tolerance: 1e-3,

        solver_name: RKV767e,
        solver: ExplicitRungeKutta::rkv767e(),
        tolerance: 1e-3,

        solver_name: RKV877e,
        solver: ExplicitRungeKutta::rkv877e(),
        tolerance: 1e-3,

        solver_name: RKV878e,
        solver: ExplicitRungeKutta::rkv878e(),
        tolerance: 1e-3,

        solver_name: RKV988e,
        solver: ExplicitRungeKutta::rkv988e(),
        tolerance: 1e-3,

        solver_name: RKV989e,
        solver: ExplicitRungeKutta::rkv989e(),
        tolerance: 1e-3,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
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
        solver: ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12),
        tolerance: 1e-3,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5(),
        tolerance: 1e-3,

        solver_name: RKF45,
        solver: ExplicitRungeKutta::rkf45(),
        tolerance: 1e-2,

        solver_name: CashKarp,
        solver: ExplicitRungeKutta::cash_karp(),
        tolerance: 1e-2,

        solver_name: RK4,
        solver: ExplicitRungeKutta::rk4(0.01),
        tolerance: 1e3,

        solver_name: ThreeEighths,
        solver: ExplicitRungeKutta::three_eighths(0.01),
        tolerance: 1e3,

        solver_name: Euler,
        solver: ExplicitRungeKutta::euler(0.01),
        tolerance: 1e-2,

        solver_name: Midpoint,
        solver: ExplicitRungeKutta::midpoint(0.01),
        tolerance: 1e-2,

        solver_name: Heun,
        solver: ExplicitRungeKutta::heun(0.01),
        tolerance: 1e-2,

        solver_name: Ralston,
        solver: ExplicitRungeKutta::ralston(0.01),
        tolerance: 1e-2,

        solver_name: APCF4,
        solver: AdamsPredictorCorrector::f4(0.01),
        tolerance: 1e-2,

        solver_name: APCV4,
        solver: AdamsPredictorCorrector::v4(),
        tolerance: 1e-2,

        solver_name: RKV65,
        solver: ExplicitRungeKutta::rkv655e(),
        tolerance: 1e-3,

        solver_name: RKV766e,
        solver: ExplicitRungeKutta::rkv766e(),
        tolerance: 1e-3,

        solver_name: RKV767e,
        solver: ExplicitRungeKutta::rkv767e(),
        tolerance: 1e-3,

        solver_name: RKV877e,
        solver: ExplicitRungeKutta::rkv877e(),
        tolerance: 1e-3,

        solver_name: RKV878e,
        solver: ExplicitRungeKutta::rkv878e(),
        tolerance: 1e-3,

        solver_name: RKV988e,
        solver: ExplicitRungeKutta::rkv988e(),
        tolerance: 1e-3,

        solver_name: RKV989e,
        solver: ExplicitRungeKutta::rkv989e(),
        tolerance: 1e-3,

        // Implicit methods

        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(0.01),
        tolerance: 1e-3,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4(),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6(),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
        tolerance: 1e-3,

        // Diagonally implicit methods

        solver_name: SDIRK21,
        solver: DiagonallyImplicitRungeKutta::sdirk21(),
        tolerance: 1e-3,

        solver_name: ESDIRK33,
        solver: DiagonallyImplicitRungeKutta::esdirk33(0.01),
        tolerance: 1e-3
    }

    test_ode! {
        system_name: robertson_stiff_problem,
        ode: RobertsonProblem,
        t0: 0.0,
        tf: 0.4,
        y0: vector![1.0, 0.0, 0.0],
        expected_result: vector![0.9851721139, 3.3863953790e-05, 0.0147940222],

        // Implicit methods
        solver_name: CrankNicolson,
        solver: ImplicitRungeKutta::crank_nicolson(0.001),
        tolerance: 1e-2,

        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4().rtol(1e-6).atol(1e-8),
        tolerance: 1e-3,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6().rtol(1e-6).atol(1e-8),
        tolerance: 1e-3,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
        tolerance: 1e-3,

        // DIRK methods
        solver_name: SDIRK21,
        solver: DiagonallyImplicitRungeKutta::sdirk21().rtol(1e-6).atol(1e-8),
        tolerance: 1e-2,

        solver_name: ESDIRK33,
        solver: DiagonallyImplicitRungeKutta::esdirk33(0.001),
        tolerance: 1e-2,

        solver_name: ESDIRK324L2SA,
        solver: DiagonallyImplicitRungeKutta::esdirk324l2sa().rtol(1e-6).atol(1e-8),
        tolerance: 1e-2,

        solver_name: Kvaerno423,
        solver: DiagonallyImplicitRungeKutta::kvaerno423().rtol(1e-6).atol(1e-8),
        tolerance: 1e-2,

        // Explicit methods
        solver_name: DOP853,
        solver: ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12),
        tolerance: 1e-1,

        solver_name: DOPRI5,
        solver: ExplicitRungeKutta::dopri5().rtol(1e-8).atol(1e-10),
        tolerance: 5e-1
    }

    test_ode! {
        system_name: hires_problem,
        ode: HiresProblem,
        t0: 0.0,
        tf: 100.0,
        y0: vector![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
        expected_result: vector![4.5208593641e-03, 8.8390563234e-04, 7.9719428657e-04, 7.8113260614e-03, 1.3238525410e-01, 5.3016769232e-01, 5.6313397578e-03, 6.8660242157e-05],

        // Implicit methods
        solver_name: GaussLegendre4,
        solver: ImplicitRungeKutta::gauss_legendre_4().rtol(1e-4).atol(1e-6),
        tolerance: 1e-1,

        solver_name: GaussLegendre6,
        solver: ImplicitRungeKutta::gauss_legendre_6().rtol(1e-4).atol(1e-6),
        tolerance: 1e-1,

        solver_name: Radau5,
        solver: ImplicitRungeKutta::radau5(),
        tolerance: 1e-1,

        // High-order DIRK methods
        solver_name: ESDIRK324L2SA,
        solver: DiagonallyImplicitRungeKutta::esdirk324l2sa().rtol(1e-4).atol(1e-6),
        tolerance: 1e-1,

        solver_name: Kvaerno423,
        solver: DiagonallyImplicitRungeKutta::kvaerno423().rtol(1e-4).atol(1e-6),
        tolerance: 1e-1
    }
}
