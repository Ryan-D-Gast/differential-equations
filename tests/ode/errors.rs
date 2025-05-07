//! Suite of test cases for NumericalMethods error handling

use differential_equations::{
    ControlFlag, Error, Status,
    ode::{
        ODE, ODEProblem,
        methods::{
            adams::{APCF4, APCV4},
            runge_kutta::{
                explicit::{DOP853, DOPRI5, Euler, RK4, RKF, RKV65, RKV98},
                implicit::{CrankNicolson, GaussLegendre6},
            },
        },
    },
};
use nalgebra::{SVector, vector};

struct SimpleODE;

impl ODE<f64, SVector<f64, 1>> for SimpleODE {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = y[0];
    }

    fn event(&self, t: f64, _y: &SVector<f64, 1>) -> ControlFlag {
        if t == 10.0 {
            ControlFlag::Terminate("Initial condition trigger".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}

/// Macro for testing cases where we expect a Error to be returned
macro_rules! test_solver_error {
    (
        test_name: $test_name:ident,
        ode: $system:expr,
        t0: $t0:expr,
        tf: $tf:expr,
        y0: $y0:expr,
        expected_error: $expected_error:expr,
        $(solver_name: $solver_name:ident, solver: $solver:expr),+
    ) => {
        $(
            // Initialize the system
            let system = $system;
            let t0 = $t0;
            let tf = $tf;
            let y0 = $y0;

            let problem = ODEProblem::new(system, t0, tf, y0);
            let mut solver = $solver;

            // Solve the system
            let results = problem.solve(&mut solver);

            // Assert the result matches expected error
            match results {
                Ok(r) => {
                    panic!("Test {} {} failed: Expected error {:?} but got success with status {:?}",
                        stringify!($solver_name), stringify!($test_name), $expected_error, r.status);
                },
                Err(e) => {
                    assert_eq!(e, $expected_error,
                        "Test {} {} failed: Expected: {:?}, Got: {:?}",
                        stringify!($solver_name), stringify!($test_name), $expected_error, e);
                    println!("{} {} passed with expected error: {:?}",
                        stringify!($solver_name), stringify!($test_name), e);
                }
            }
        )+
    };
}

/// Macro for testing cases where we expect a successful solution with specific status
macro_rules! test_solver_status {
    (
        test_name: $test_name:ident,
        ode: $system:expr,
        t0: $t0:expr,
        tf: $tf:expr,
        y0: $y0:expr,
        expected_status: $expected_status:expr,
        $(solver_name: $solver_name:ident, solver: $solver:expr),+
    ) => {
        $(
            // Initialize the system
            let system = $system;
            let t0 = $t0;
            let tf = $tf;
            let y0 = $y0;

            let problem = ODEProblem::new(system, t0, tf, y0);
            let mut solver = $solver;

            // Solve the system
            let results = problem.solve(&mut solver);

            // Assert the result matches expected status
            match results {
                Ok(r) => {
                    let stat = r.status;
                    assert_eq!(stat, $expected_status,
                        "Test {} {} failed: Expected status: {:?}, Got: {:?}",
                        stringify!($solver_name), stringify!($test_name), $expected_status, stat);
                    println!("{} {} passed with expected status: {:?}",
                        stringify!($solver_name), stringify!($test_name), stat);
                },
                Err(e) => {
                    panic!("Test {} {} failed: Expected status {:?} but got error {:?}",
                        stringify!($solver_name), stringify!($test_name), $expected_status, e);
                }
            }
        )+
    };
}

#[test]
fn invalid_time_span() {
    test_solver_error! {
        test_name: invalid_time_span,
        ode: SimpleODE,
        t0: 0.0,
        tf: 0.0,
        y0: vector![1.0],
        expected_error: Error::<f64, SVector<f64, 1>>::BadInput { msg: "Invalid input: tf (0.0) cannot be equal to t0 (0.0)".to_string() },
        solver_name: DOP853, solver: DOP853::new(),
        solver_name: DOPRI5, solver: DOPRI5::new(),
        solver_name: RKF, solver: RKF::new().h0(0.1),
        solver_name: RK4, solver: RK4::new(0.1),
        solver_name: Euler, solver: Euler::new(0.1),
        solver_name: APCF4, solver: APCF4::new(0.1),
        solver_name: APCV4, solver: APCV4::new().h0(0.1),
        solver_name: RKV65, solver: RKV65::new().h0(0.1),
        solver_name: RKV98, solver: RKV98::new().h0(0.1),
        solver_name: CrankNicolson, solver: CrankNicolson::new(0.1),
        solver_name: GaussLegendre6, solver: GaussLegendre6::new().h0(0.1)
    }
}

#[test]
fn initial_step_size_too_big() {
    test_solver_error! {
        test_name: initial_step_size_too_big,
        ode: SimpleODE,
        t0: 0.0,
        tf: 1.0,
        y0: vector![1.0],
        expected_error: Error::<f64, SVector<f64, 1>>::BadInput { msg: "Invalid input: Absolute value of initial step size (10.0) must be less than or equal to the absolute value of the integration interval (tf - t0 = 1.0)".to_string() },
        solver_name: DOP853, solver: DOP853::new().h0(10.0),
        solver_name: DOPRI5, solver: DOPRI5::new().h0(10.0),
        solver_name: RKF, solver: RKF::new().h0(10.0),
        solver_name: RK4, solver: RK4::new(10.0),
        solver_name: Euler, solver: Euler::new(10.0),
        solver_name: APCF4, solver: APCF4::new(10.0),
        solver_name: APCV4, solver: APCV4::new().h0(10.0),
        solver_name: RKV65, solver: RKV65::new().h0(10.0),
        solver_name: RKV98, solver: RKV98::new().h0(10.0),
        solver_name: CrankNicolson, solver: CrankNicolson::new(10.0),
        solver_name: GaussLegendre6, solver: GaussLegendre6::new().h0(10.0)
    }
}

#[test]
fn terminate_initial_conditions_trigger() {
    test_solver_status! {
        test_name: terminate_initial_conditions_trigger,
        ode: SimpleODE,
        t0: 10.0,
        tf: 20.0,
        y0: vector![1.0],
        expected_status: Status::<f64, SVector<f64, 1>, String>::Interrupted("Initial condition trigger".to_string()),
        solver_name: DOP853, solver: DOP853::new(),
        solver_name: DOPRI5, solver: DOPRI5::new(),
        solver_name: RKF, solver: RKF::new().h0(0.1),
        solver_name: RK4, solver: RK4::new(0.1),
        solver_name: Euler, solver: Euler::new(0.1),
        solver_name: APCF4, solver: APCF4::new(0.1),
        solver_name: APCV4, solver: APCV4::new().h0(0.1),
        solver_name: RKV65, solver: RKV65::new().h0(0.1),
        solver_name: RKV98, solver: RKV98::new().h0(0.1),
        solver_name: CrankNicolson, solver: CrankNicolson::new(0.1),
        solver_name: GaussLegendre6, solver: GaussLegendre6::new().h0(0.1)
    }
}
