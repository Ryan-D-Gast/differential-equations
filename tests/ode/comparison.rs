//! Compares the performance of solvers by the statistics, i.e. number of steps, function evaluations, etc.

use super::systems;
use differential_equations::ode::IVP;
use differential_equations::ode::methods::{
    APCV4, DOP853, DOPRI5, RKF, RKV65, RKV98,
};
use nalgebra::SVector;
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};
use systems::{VanDerPolOscillator, LorenzSystem, BrusselatorSystem, Cr3bp};

struct TestStatistics<const N: usize> {
    name: String,
    steps: usize,
    evals: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    tolerance: f64, // Added tolerance field
    yf: SVector<f64, N>,
}

macro_rules! generate_error_vs_steps_lorenz {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_lorenz() {
            let mut statistics = Vec::new();

            // Lorenz System (chaotic)
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 3>::new(1.0, 1.0, 1.0);
            let sigma = 10.0;
            let rho = 28.0;
            let beta = 8.0 / 3.0;
            let system = LorenzSystem { sigma, rho, beta };
            let ivp = IVP::new(system, t0, tf, y0);
            
            // Get reference solution with very high accuracy
            let mut reference_solver = RKV98::new().rtol(1e-14).atol(1e-14);
            let reference_sol = ivp.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();
            
            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];
            
            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = ivp.solve(&mut solver).unwrap();
                    
                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        steps: sol.steps,
                        evals: sol.evals,
                        accepted_steps: sol.accepted_steps,
                        rejected_steps: sol.rejected_steps,
                        tolerance: tol,
                        yf: sol.y.last().unwrap().clone(),
                    });
                }
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/ode/comparison/").expect("Failed to create directory");
            let path = Path::new("target/tests/ode/comparison/error_vs_evals_lorenz.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "NumericalMethod,Tolerance,Steps,Evals,Accepted,Rejected,Global Error").unwrap();

            for stats in statistics {
                // Calculate error compared to reference solution
                let error = (stats.yf - &reference_yf).norm();

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{}",
                    stats.name, 
                    stats.tolerance,
                    stats.steps, 
                    stats.evals, 
                    stats.accepted_steps, 
                    stats.rejected_steps, 
                    error
                ).unwrap();
            }
        }
    };
}

macro_rules! generate_error_vs_steps_vanderpol {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_vanderpol() {
            let mut statistics = Vec::new();

            // Van der Pol Oscillator
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 2>::new(2.0, 0.0);
            let mu = 5.0; // Higher values make the problem more stiff
            let system = VanDerPolOscillator { mu };
            let ivp = IVP::new(system, t0, tf, y0);
            
            // Get reference solution with very high accuracy
            let mut reference_solver = RKV98::new().rtol(1e-14).atol(1e-14);
            let reference_sol = ivp.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();
            
            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];
            
            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = ivp.solve(&mut solver).unwrap();
                    
                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        steps: sol.steps,
                        evals: sol.evals,
                        accepted_steps: sol.accepted_steps,
                        rejected_steps: sol.rejected_steps,
                        tolerance: tol,
                        yf: sol.y.last().unwrap().clone(),
                    });
                }
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/ode/comparison/").expect("Failed to create directory");
            let path = Path::new("target/tests/ode/comparison/error_vs_evals_vanderpol.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "NumericalMethod,Tolerance,Steps,Evals,Accepted,Rejected,Global Error").unwrap();

            for stats in statistics {
                // Calculate error compared to reference solution
                let error = (stats.yf - &reference_yf).norm();

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{}",
                    stats.name, 
                    stats.tolerance,
                    stats.steps, 
                    stats.evals, 
                    stats.accepted_steps, 
                    stats.rejected_steps, 
                    error
                ).unwrap();
            }
        }
    };
}

macro_rules! generate_error_vs_steps_brusselator {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_brusselator() {
            let mut statistics = Vec::new();

            // Brusselator System - exhibits limit cycles and is a standard test for stiff solvers
            let t0 = 0.0;
            let tf = 20.0;
            let y0 = SVector::<f64, 2>::new(1.0, 1.0);
            let a = 1.0;
            let b = 3.0;
            let system = BrusselatorSystem { a, b };
            let ivp = IVP::new(system, t0, tf, y0);
            
            // Get reference solution with very high accuracy
            let mut reference_solver = RKV98::new().rtol(1e-14).atol(1e-14);
            let reference_sol = ivp.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();
            
            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];
            
            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = ivp.solve(&mut solver).unwrap();
                    
                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        steps: sol.steps,
                        evals: sol.evals,
                        accepted_steps: sol.accepted_steps,
                        rejected_steps: sol.rejected_steps,
                        tolerance: tol,
                        yf: sol.y.last().unwrap().clone(),
                    });
                }
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/ode/comparison/").expect("Failed to create directory");
            let path = Path::new("target/tests/ode/comparison/error_vs_evals_brusselator.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "NumericalMethod,Tolerance,Steps,Evals,Accepted,Rejected,Global Error").unwrap();

            for stats in statistics {
                // Calculate error compared to reference solution
                let error = (stats.yf - &reference_yf).norm();

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{}",
                    stats.name, 
                    stats.tolerance,
                    stats.steps, 
                    stats.evals, 
                    stats.accepted_steps, 
                    stats.rejected_steps, 
                    error
                ).unwrap();
            }
        }
    };
}

macro_rules! generate_error_vs_steps_cr3bp {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_cr3bp() {
            let mut statistics = Vec::new();

            // Circular Restricted Three Body Problem
            let t0 = 0.0;
            let tf = 10.0;
            // Initial state vector: position (x,y,z) and velocity (vx,vy,vz)
            let y0 = SVector::<f64, 6>::new(
                // 9:2 L2 Southern NRHO orbit
                1.021881345465263,
                0.0,
                -0.182000000000000, // Position
                0.0,
                -0.102950816739606,
                0.0 // Velocity
            );
            let mu = 0.012150585609624; // Earth-Moon system
            let system = Cr3bp { mu };
            let ivp = IVP::new(system, t0, tf, y0);
            
            // Get reference solution with very high accuracy
            let mut reference_solver = RKV98::new().rtol(1e-14).atol(1e-14);
            let reference_sol = ivp.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();
            
            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];
            
            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = ivp.solve(&mut solver).unwrap();
                    
                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        steps: sol.steps,
                        evals: sol.evals,
                        accepted_steps: sol.accepted_steps,
                        rejected_steps: sol.rejected_steps,
                        tolerance: tol,
                        yf: sol.y.last().unwrap().clone(),
                    });
                }
            )+

            // Write Statistics to CSV
            fs::create_dir_all("target/tests/ode/comparison/").expect("Failed to create directory");
            let path = Path::new("target/tests/ode/comparison/error_vs_evals_cr3bp.csv");
            let mut file = File::create(path).expect("Failed to create file");
            writeln!(file, "NumericalMethod,Tolerance,Steps,Evals,Accepted,Rejected,Global Error").unwrap();

            for stats in statistics {
                // Calculate error compared to reference solution
                let error = (stats.yf - &reference_yf).norm();

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{}",
                    stats.name, 
                    stats.tolerance,
                    stats.steps, 
                    stats.evals, 
                    stats.accepted_steps, 
                    stats.rejected_steps, 
                    error
                ).unwrap();
            }
        }
    };
}

// Ignored by default due to large cost and doesn't assert anything, here for creating plots
#[test] 
#[ignore] // Run via `cargo test --test comparison -- --ignored` to include this test
fn error_vs_evals() {
    generate_error_vs_steps_lorenz! {
        DOP853, |tol| DOP853::new().rtol(tol).atol(tol),
        DOPRI5, |tol| DOPRI5::new().rtol(tol).atol(tol),
        RKF, |tol| RKF::new().rtol(tol).atol(tol),
        APCV4, |tol| APCV4::new().tol(tol),
        RKV65, |tol| RKV65::new().rtol(tol).atol(tol),
        RKV98, |tol| RKV98::new().rtol(tol).atol(tol)
    }

    generate_error_vs_steps_vanderpol! {
        DOP853, |tol| DOP853::new().rtol(tol).atol(tol),
        DOPRI5, |tol| DOPRI5::new().rtol(tol).atol(tol),
        RKF, |tol| RKF::new().rtol(tol).atol(tol),
        APCV4, |tol| APCV4::new().tol(tol),
        RKV65, |tol| RKV65::new().rtol(tol).atol(tol),
        RKV98, |tol| RKV98::new().rtol(tol).atol(tol)
    }
    
    generate_error_vs_steps_brusselator! {
        DOP853, |tol| DOP853::new().rtol(tol).atol(tol),
        DOPRI5, |tol| DOPRI5::new().rtol(tol).atol(tol),
        RKF, |tol| RKF::new().rtol(tol).atol(tol),
        APCV4, |tol| APCV4::new().tol(tol),
        RKV65, |tol| RKV65::new().rtol(tol).atol(tol),
        RKV98, |tol| RKV98::new().rtol(tol).atol(tol)
    }
    
    generate_error_vs_steps_cr3bp! {
        DOP853, |tol| DOP853::new().rtol(tol).atol(tol),
        DOPRI5, |tol| DOPRI5::new().rtol(tol).atol(tol),
        RKF, |tol| RKF::new().rtol(tol).atol(tol),
        APCV4, |tol| APCV4::new().tol(tol),
        RKV65, |tol| RKV65::new().rtol(tol).atol(tol),
        RKV98, |tol| RKV98::new().rtol(tol).atol(tol)
    }

    error_vs_steps_lorenz();
    error_vs_steps_vanderpol();
    error_vs_steps_brusselator();
    error_vs_steps_cr3bp();
}
