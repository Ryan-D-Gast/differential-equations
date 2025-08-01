//! Compares the performance of solvers by the statistics, i.e. number of steps, function evaluations, etc.

use super::systems;
use differential_equations::{
    methods::ExplicitRungeKutta,
    ode::ODEProblem,
};
use nalgebra::SVector;
use std::fs;
use systems::{BrusselatorSystem, Cr3bp, LorenzSystem, VanDerPolOscillator};
use quill::*;

struct TestStatistics<const N: usize> {
    name: String,
    evals: usize,
    error: f64,
}

macro_rules! generate_error_vs_steps_lorenz {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_lorenz() -> Vec<TestStatistics<3>> {
            let mut statistics = Vec::new();

            // Lorenz System (chaotic)
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 3>::new(1.0, 1.0, 1.0);
            let sigma = 10.0;
            let rho = 28.0;
            let beta = 8.0 / 3.0;
            let system = LorenzSystem { sigma, rho, beta };
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Get reference solution with very high accuracy
            let mut reference_solver = ExplicitRungeKutta::rkv988e().rtol(1e-14).atol(1e-14);
            let reference_sol = problem.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();

            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];

            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = problem.solve(&mut solver).unwrap();
                    let yf = sol.y.last().unwrap().clone();
                    let error = (yf - &reference_yf).norm();

                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        evals: sol.evals.function,
                        error,
                    });
                }
            )+

            statistics
        }
    };
}

macro_rules! generate_error_vs_steps_vanderpol {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_vanderpol() -> Vec<TestStatistics<2>> {
            let mut statistics = Vec::new();

            // Van der Pol Oscillator
            let t0 = 0.0;
            let tf = 10.0;
            let y0 = SVector::<f64, 2>::new(2.0, 0.0);
            let mu = 5.0; // Higher values make the problem more stiff
            let system = VanDerPolOscillator { mu };
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Get reference solution with very high accuracy
            let mut reference_solver = ExplicitRungeKutta::rkv988e().rtol(1e-14).atol(1e-14);
            let reference_sol = problem.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();

            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];

            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = problem.solve(&mut solver).unwrap();
                    let yf = sol.y.last().unwrap().clone();
                    let error = (yf - &reference_yf).norm();

                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        evals: sol.evals.function,
                        error,
                    });
                }
            )+

            statistics
        }
    };
}

macro_rules! generate_error_vs_steps_brusselator {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_brusselator() -> Vec<TestStatistics<2>> {
            let mut statistics = Vec::new();

            // Brusselator System - exhibits limit cycles and is a standard test for stiff solvers
            let t0 = 0.0;
            let tf = 20.0;
            let y0 = SVector::<f64, 2>::new(1.0, 1.0);
            let a = 1.0;
            let b = 3.0;
            let system = BrusselatorSystem { a, b };
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Get reference solution with very high accuracy
            let mut reference_solver = ExplicitRungeKutta::rkv988e().rtol(1e-14).atol(1e-14);
            let reference_sol = problem.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();

            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];

            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = problem.solve(&mut solver).unwrap();
                    let yf = sol.y.last().unwrap().clone();
                    let error = (yf - &reference_yf).norm();

                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        evals: sol.evals.function,
                        error,
                    });
                }
            )+

            statistics
        }
    };
}

macro_rules! generate_error_vs_steps_cr3bp {
    (
        $(
            $solver_name:ident, $solver_generator:expr
        ),+
    ) => {
        fn error_vs_steps_cr3bp() -> Vec<TestStatistics<6>> {
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
            let problem = ODEProblem::new(system, t0, tf, y0);

            // Get reference solution with very high accuracy
            let mut reference_solver = ExplicitRungeKutta::rkv988e().rtol(1e-14).atol(1e-14);
            let reference_sol = problem.solve(&mut reference_solver).unwrap();
            let reference_yf = reference_sol.y.last().unwrap().clone();

            // Test each solver with different tolerance values
            let tolerance_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12];

            $(
                for &tol in &tolerance_values {
                    let mut solver = $solver_generator(tol);
                    let sol = problem.solve(&mut solver).unwrap();
                    let yf = sol.y.last().unwrap().clone();
                    let error = (yf - &reference_yf).norm();

                    statistics.push(TestStatistics {
                        name: stringify!($solver_name).to_string(),
                        evals: sol.evals.function,
                        error,
                    });
                }
            )+

            statistics
        }
    };
}

fn create_error_vs_evals_plot<const N: usize>(statistics: &[TestStatistics<N>], problem_name: &str) {
    // Create directory for plots
    fs::create_dir_all("target/tests/ode/comparison/plots").expect("Failed to create directory");
    
    // Get unique solver names
    let mut solver_names: Vec<String> = statistics.iter()
        .map(|s| s.name.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    solver_names.sort();
    
    // Define colors for different solvers
    let colors = [
        Color::Blue,
        Color::Red,
        Color::Green,
        Color::Orange,
        Color::Brown,
        Color::Pink,
    ];
    
    // Create series for each solver
    let mut series_list: [Option<Series<f64>>; 7] = [None, None, None, None, None, None, None];
    
    // Create display names with extra spaces for better plot formatting
    let display_names: Vec<String> = solver_names.iter()
        .map(|name| format!("{}   ", name))
        .collect();

    for (i, solver_name) in solver_names.iter().enumerate() {
        let solver_data: Vec<_> = statistics.iter()
            .filter(|s| s.name == *solver_name)
            .collect();
        
        // Sort by number of evaluations
        let mut data_points: Vec<(f64, f64)> = solver_data.iter()
            .map(|s| (s.evals as f64, s.error))
            .collect();
        data_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let series = Series::builder()
            .name(&display_names[i])
            .color(colors[i % colors.len()].clone())
            .data(data_points)
            .build();
        
        series_list[i] = Some(series);
    }
    
    // Convert to actual array of Series
    let final_series_list: [Series<f64>; 6] = [
        series_list[0].take().unwrap(),
        series_list[1].take().unwrap(),
        series_list[2].take().unwrap(),
        series_list[3].take().unwrap(),
        series_list[4].take().unwrap(),
        series_list[5].take().unwrap(),
    ];
    
    // Capitalize first letter for the title
    let title_problem_name = format!("{}{}", 
        problem_name.chars().next().unwrap().to_uppercase(),
        &problem_name[1..]
    );
    
    // Create and save the plot
    Plot::builder()
        .title(&format!("{} - Error vs Function Evaluations", title_problem_name))
        .x_label("Number of Function Evaluations")
        .y_label("Error")
        .y_scale(Scale::Log)
        .x_scale(Scale::Log)
        .minor_grid(MinorGrid::Both)
        .legend(Legend::TopRightInside)
        .data(final_series_list)
        .build()
        .to_svg(&format!("target/tests/ode/comparison/plots/error_vs_evals_{}.svg", problem_name))
        .expect("Failed to save plot as SVG");
}

// Ignored by default due to large cost and doesn't assert anything, here for creating plots
#[test]
#[ignore] // Run via `cargo test --test comparison -- --ignored` to include this test
fn test_error_vs_evals() {
    generate_error_vs_steps_lorenz! {
        DOP853, |tol| ExplicitRungeKutta::dop853().rtol(tol).atol(tol),
        DOPRI5, |tol| ExplicitRungeKutta::dopri5().rtol(tol).atol(tol),
        RKF, |tol| ExplicitRungeKutta::rkf45().rtol(tol).atol(tol),
        RKV65, |tol| ExplicitRungeKutta::rkv656e().rtol(tol).atol(tol),
        RKV87, |tol| ExplicitRungeKutta::rkv877e().rtol(tol).atol(tol),
        RKV98, |tol| ExplicitRungeKutta::rkv988e().rtol(tol).atol(tol)
    }

    generate_error_vs_steps_vanderpol! {
        DOP853, |tol| ExplicitRungeKutta::dop853().rtol(tol).atol(tol),
        DOPRI5, |tol| ExplicitRungeKutta::dopri5().rtol(tol).atol(tol),
        RKF, |tol| ExplicitRungeKutta::rkf45().rtol(tol).atol(tol),
        RKV65, |tol| ExplicitRungeKutta::rkv656e().rtol(tol).atol(tol),
        RKV87, |tol| ExplicitRungeKutta::rkv877e().rtol(tol).atol(tol),
        RKV98, |tol| ExplicitRungeKutta::rkv988e().rtol(tol).atol(tol)
    }

    generate_error_vs_steps_brusselator! {
        DOP853, |tol| ExplicitRungeKutta::dop853().rtol(tol).atol(tol),
        DOPRI5, |tol| ExplicitRungeKutta::dopri5().rtol(tol).atol(tol),
        RKF, |tol| ExplicitRungeKutta::rkf45().rtol(tol).atol(tol),
        RKV65, |tol| ExplicitRungeKutta::rkv656e().rtol(tol).atol(tol),
        RKV87, |tol| ExplicitRungeKutta::rkv877e().rtol(tol).atol(tol),
        RKV98, |tol| ExplicitRungeKutta::rkv988e().rtol(tol).atol(tol)
    }

    generate_error_vs_steps_cr3bp! {
        DOP853, |tol| ExplicitRungeKutta::dop853().rtol(tol).atol(tol),
        DOPRI5, |tol| ExplicitRungeKutta::dopri5().rtol(tol).atol(tol),
        RKF, |tol| ExplicitRungeKutta::rkf45().rtol(tol).atol(tol),
        RKV65, |tol| ExplicitRungeKutta::rkv656e().rtol(tol).atol(tol),
        RKV87, |tol| ExplicitRungeKutta::rkv877e().rtol(tol).atol(tol),
        RKV98, |tol| ExplicitRungeKutta::rkv988e().rtol(tol).atol(tol)
    }

    // Generate and plot all the data
    let lorenz_stats = error_vs_steps_lorenz();
    let vanderpol_stats = error_vs_steps_vanderpol();
    let brusselator_stats = error_vs_steps_brusselator();
    let cr3bp_stats = error_vs_steps_cr3bp();

    // Create plots using quill
    create_error_vs_evals_plot(&lorenz_stats, "lorenz");
    create_error_vs_evals_plot(&vanderpol_stats, "vanderpol");
    create_error_vs_evals_plot(&brusselator_stats, "brusselator");
    create_error_vs_evals_plot(&cr3bp_stats, "cr3bp");
}