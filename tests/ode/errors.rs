//! Suite of test cases for Solvers error handling

use differential_equations::ode::IVP;
use differential_equations::ode::solvers::{DOP853, RK4, RKF, Euler, APCF4, APCV4, Verner65, Verner98};
use differential_equations::ode::{EventAction, SolverStatus};
use differential_equations::ode::ODE;
use nalgebra::{SVector, vector};

struct SimpleODE;

impl ODE<f64, 1, 1> for SimpleODE {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = y[0];
    }

    fn event(&self, t: f64, _y: &SVector<f64, 1>, _dydt: &SVector<f64, 1>) -> EventAction {
        if t == 10.0 {
            EventAction::Terminate("Initial condition trigger".to_string())
        } else {
            EventAction::Continue
        }
    }
}

macro_rules! test_error_ode {
    (
        test_name: $test_name:ident,
        ode: $system:expr,
        t0: $t0:expr,
        tf: $tf:expr,
        y0: $y0:expr,
        expected_result: $expected_result:expr,
        $(solver_name: $solver_name:ident, solver: $solver:expr),+
    ) => {
        $(
            // Initialize the system
            let system = $system;

            // Set initial conditions
            let t0 = $t0;
            let tf = $tf;
            let y0 = $y0;

            // Create Initial Value Problem (IVP) for the system
            let ivp = IVP::new(system, t0, tf, y0);

            // Initialize the solver
            let mut solver = $solver;

            // Solve the system
            let results = ivp.solve(&mut solver);

            // Assert the result
            match results {
                Ok(result) => {
                    let status = result.status;
                    assert_eq!(status, $expected_result, "Test {} {} failed: Expected: {:?}, Got: {:?}", stringify!($solver_name), stringify!($test_name), $expected_result, status);
                },
                Err(e) => {
                    //println!("Test {} {} failed with error: {:?}", stringify!($solver_name), stringify!($test_name), e);
                    assert_eq!(e, $expected_result, "Test {:?} {:?} failed: Expected: {:?}, Got: {:?}", stringify!($solver_name), stringify!($test_name), $expected_result, e);
                }
            }
            println!("{} {} passed", stringify!($solver_name), stringify!($test_name));
        )+
    };
}

#[test]
fn invalid_time_span() {
    test_error_ode! {
        test_name: invalid_time_span,
        ode: SimpleODE,
        t0: 0.0,
        tf: 0.0,
        y0: vector![1.0],
        expected_result: SolverStatus::<f64, 1, 1, String>::BadInput("Invalid input: tf (0.0) cannot be equal to t0 (0.0)".to_string()),
        solver_name: DOP853, solver: DOP853::new(),
        solver_name: RKF, solver: RKF::new(0.1),
        solver_name: RK4, solver: RK4::new(0.1),
        solver_name: Euler, solver: Euler::new(0.1),
        solver_name: APCF4, solver: APCF4::new(0.1),
        solver_name: APCV4, solver: APCV4::new(0.1),
        solver_name: Verner65, solver: Verner65::new(0.1),
        solver_name: Verner98, solver: Verner98::new(0.1)
    }
}

#[test]
fn initial_step_size_too_big() {
    test_error_ode! {
        test_name: initial_step_size_too_big,
        ode: SimpleODE,
        t0: 0.0,
        tf: 1.0,
        y0: vector![1.0],
        expected_result: SolverStatus::<f64, 1, 1, String>::BadInput("Invalid input: Absolute value of initial step size (10.0) must be less than or equal to the absolute value of the integration interval (tf - t0 = 1.0)".to_string()),
        solver_name: DOP853, solver: DOP853::new().h0(10.0),
        solver_name: RKF, solver: RKF::new(10.0),
        solver_name: RK4, solver: RK4::new(10.0),
        solver_name: Euler, solver: Euler::new(10.0),
        solver_name: APCF4, solver: APCF4::new(10.0),
        solver_name: APCV4, solver: APCV4::new(10.0),
        solver_name: Verner65, solver: Verner65::new(10.0),
        solver_name: Verner98, solver: Verner98::new(10.0)
    }
}

#[test]
fn terminate_initial_conditions_trigger() {
    test_error_ode! {
        test_name: terminate_initial_conditions_trigger,
        ode: SimpleODE,
        t0: 10.0,
        tf: 20.0,
        y0: vector![1.0],
        expected_result: SolverStatus::<f64, 1, 1, String>::Interrupted("Initial condition trigger".to_string()),
        solver_name: DOP853, solver: DOP853::new(),
        solver_name: RKF, solver: RKF::new(0.1),
        solver_name: RK4, solver: RK4::new(0.1),
        solver_name: Euler, solver: Euler::new(0.1),
        solver_name: APCF4, solver: APCF4::new(0.1),
        solver_name: APCV4, solver: APCV4::new(0.1),
        solver_name: Verner65, solver: Verner65::new(0.1),
        solver_name: Verner98, solver: Verner98::new(0.1)
    }
}