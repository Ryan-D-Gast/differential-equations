//! Suite of test cases for numerical method error handling

use differential_equations::{
    control::ControlFlag,
    error::Error,
    methods::{AdamsPredictorCorrector, ExplicitRungeKutta, ImplicitRungeKutta},
    ode::{ODE, ODEProblem},
    status::Status,
};
use nalgebra::{SVector, vector};

struct SimpleODE;

impl ODE<f64, SVector<f64, 1>> for SimpleODE {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = y[0];
    }

    fn event(&self, t: f64, _y: &SVector<f64, 1>) -> ControlFlag<f64, SVector<f64, 1>> {
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
                    // Compare only the error variant to avoid brittle string checks
                    use std::mem::discriminant;
                    let got_kind = discriminant(&e);
                    let expected_kind = discriminant(&$expected_error);
                    assert_eq!(got_kind, expected_kind,
                        "Test {} {} failed: Expected error kind {:?}, Got {:?}",
                        stringify!($solver_name), stringify!($test_name), $expected_error, e);

                    // For BadInput, ensure message conveys invalid input without depending on exact wording
                    if let (Error::BadInput { msg: got_msg },
                            Error::BadInput { .. }) = (&e, &$expected_error) {
                        assert!(got_msg.starts_with("Invalid input"),
                            "Test {} {} failed: Expected BadInput message to start with 'Invalid input', got: {}",
                            stringify!($solver_name), stringify!($test_name), got_msg);
                    }

                    println!("{} {} passed with expected error kind: {:?}",
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
        solver_name: DOP853, solver: ExplicitRungeKutta::dop853(),
        solver_name: DOPRI5, solver: ExplicitRungeKutta::dopri5(),
        solver_name: RKF, solver: ExplicitRungeKutta::rkf45().h0(0.1),
        solver_name: RK4, solver: ExplicitRungeKutta::rk4(0.1),
        solver_name: Euler, solver: ExplicitRungeKutta::euler(0.1),
        solver_name: APCF4, solver: AdamsPredictorCorrector::f4(0.1),
        solver_name: APCV4, solver: AdamsPredictorCorrector::v4().h0(0.1),
        solver_name: RKV65, solver: ExplicitRungeKutta::rkv655e().h0(0.1),
        solver_name: RKV98, solver: ExplicitRungeKutta::rkv988e().h0(0.1),
        solver_name: CrankNicolson, solver: ImplicitRungeKutta::crank_nicolson(0.1),
        solver_name: GaussLegendre6, solver: ImplicitRungeKutta::gauss_legendre_6().h0(0.1)
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
        solver_name: DOP853, solver: ExplicitRungeKutta::dop853().h0(10.0),
        solver_name: DOPRI5, solver: ExplicitRungeKutta::dopri5().h0(10.0),
        solver_name: RKF, solver: ExplicitRungeKutta::rkf45().h0(10.0),
        solver_name: RK4, solver: ExplicitRungeKutta::rk4(10.0),
        solver_name: Euler, solver: ExplicitRungeKutta::euler(10.0),
        solver_name: APCF4, solver: AdamsPredictorCorrector::f4(10.0),
        solver_name: APCV4, solver: AdamsPredictorCorrector::v4().h0(10.0),
        solver_name: RKV65, solver: ExplicitRungeKutta::rkv655e().h0(10.0),
        solver_name: RKV98, solver: ExplicitRungeKutta::rkv988e().h0(10.0),
        solver_name: CrankNicolson, solver: ImplicitRungeKutta::crank_nicolson(10.0),
        solver_name: GaussLegendre6, solver: ImplicitRungeKutta::gauss_legendre_6().h0(10.0)
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
        solver_name: DOP853, solver: ExplicitRungeKutta::dop853(),
        solver_name: DOPRI5, solver: ExplicitRungeKutta::dopri5(),
        solver_name: RKF, solver: ExplicitRungeKutta::rkf45().h0(0.1),
        solver_name: RK4, solver: ExplicitRungeKutta::rk4(0.1),
        solver_name: Euler, solver: ExplicitRungeKutta::euler(0.1),
        solver_name: APCF4, solver: AdamsPredictorCorrector::f4(0.1),
        solver_name: APCV4, solver: AdamsPredictorCorrector::v4().h0(0.1),
        solver_name: RKV65, solver: ExplicitRungeKutta::rkv655e().h0(0.1),
        solver_name: RKV98, solver: ExplicitRungeKutta::rkv988e().h0(0.1),
        solver_name: CrankNicolson, solver: ImplicitRungeKutta::crank_nicolson(0.1),
        solver_name: GaussLegendre6, solver: ImplicitRungeKutta::gauss_legendre_6().h0(0.1)
    }
}
