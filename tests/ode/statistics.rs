//! Compares the performance of solvers by the statistics, i.e. number of steps, function evaluations, etc.

use super::systems;
use systems::{HarmonicOscillator, LogisticEquation};
use differential_equations::ode::IVP;
use differential_equations::ode::solvers::{DOP853, RK4, RKF, Euler, APCF4, APCV4, RKV65, RKV98};
use std::{
    fs::{self, File},
    io::Write,
    path::Path
};
use nalgebra::SVector;

struct TestStatistics<const N: usize> {
    name: String,
    steps: usize,
    evals: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    yf: SVector<f64, N>,
}

macro_rules! generate_solver_statistics_harmonic {
    (
        $(
            $solver_name:ident, $solver:expr
        ),+
    ) => {
        fn compare_solvers_harmonic() {
            let mut statistics = Vec::new();

            // Harmonic Oscillator ODE
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 2>::new(1.0, 0.0);
            let k = 1.0;
            let system = HarmonicOscillator { k };
            let ivp = IVP::new(system, t0, tf, y0);

            $(
                let mut solver = $solver;
                let sol = ivp.solve(&mut solver).unwrap();

                statistics.push(TestStatistics {
                    name: stringify!($solver_name).to_string(),
                    steps: sol.steps,
                    evals: sol.evals,
                    accepted_steps: sol.accepted_steps,
                    rejected_steps: sol.rejected_steps,
                    yf: sol.y.last().unwrap().clone(),
                });
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/statistics/").expect("Failed to create directory");
            let path = Path::new("target/tests/statistics/solvers_harmonic.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "Solver,Steps,Evals,Accepted,Rejected,Percent Error").unwrap();

            let comparision = statistics[0].yf.clone();
            for stats in statistics {
                // Calculate the error compared to index 0
                let percent_error = (stats.yf - comparision).norm() / comparision.norm() * 100.0;

                writeln!(file, "{},{},{},{},{}, {}", stats.name, stats.steps, stats.evals, stats.accepted_steps, stats.rejected_steps, percent_error).unwrap();
            }
        }
    };
}

macro_rules! generate_solver_statistics_logistic {
    (
        $(
            $solver_name:ident, $solver:expr
        ),+
    ) => {
        fn compare_solvers_logistic() {
            let mut statistics = Vec::new();

            // Logistic Equation ODE
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 1>::new(0.1);
            let k = 1.0;
            let m = 10.0;
            let system = LogisticEquation { k, m };
            let ivp = IVP::new(system, t0, tf, y0);

            $(
                let mut solver = $solver;
                let sol = ivp.solve(&mut solver).unwrap();

                statistics.push(TestStatistics {
                    name: stringify!($solver_name).to_string(),
                    steps: sol.steps,
                    evals: sol.evals,
                    accepted_steps: sol.accepted_steps,
                    rejected_steps: sol.rejected_steps,
                    yf: sol.y.last().unwrap().clone(),
                });
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/statistics/").expect("Failed to create directory");
            let path = Path::new("target/tests/statistics/solvers_logistic.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "Solver,Steps,Evals,Accepted,Rejected,Percent Error").unwrap();

            let comparision = statistics[0].yf.clone();
            for stats in statistics {
                // Calculate the error compared to index 0
                let percent_error = (stats.yf - comparision).norm() / comparision.norm() * 100.0;

                writeln!(file, "{},{},{},{},{}, {}", stats.name, stats.steps, stats.evals, stats.accepted_steps, stats.rejected_steps, percent_error).unwrap();
            }
        }
    };
}

#[test]
fn comparision_csv() {
    generate_solver_statistics_harmonic! {
        DOP853, DOP853::new().rtol(1e-12).atol(1e-12), // Set to extremely high accuracy due to being compared against by other solvers below
        RKF, RKF::new(0.01).rtol(1e-6),
        APCV4, APCV4::new(0.01).tol(1e-6),
        RK4, RK4::new(0.01),
        Euler, Euler::new(0.01),
        APCF4, APCF4::new(0.01),
        RKV65, RKV65::new(0.01),
        RKV98, RKV98::new(0.01)
    }

    generate_solver_statistics_logistic! {
        DOP853, DOP853::new().rtol(1e-12).atol(1e-12), // Set to extremely high accuracy due to being compared against by other solvers below
        RKF, RKF::new(0.01).rtol(1e-6),
        APCV4, APCV4::new(0.01).tol(1e-6),
        RK4, RK4::new(0.01),
        Euler, Euler::new(0.01),
        APCF4, APCF4::new(0.01),
        RKV65, RKV65::new(0.01),
        RKV98, RKV98::new(0.01)
    }

    compare_solvers_harmonic();
    compare_solvers_logistic();
}