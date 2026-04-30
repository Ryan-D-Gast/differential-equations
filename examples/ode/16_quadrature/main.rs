//! Example 16: Numerical Quadrature via State Augmentation
//!
//! This example demonstrates how to compute a numerical integral alongside an ODE
//! by encoding the quadrature as an additional equation in the system. No special
//! quadrature solver or callback is needed — the existing adaptive step-size control
//! and per-component tolerances handle everything.
//!
//! We solve the exponential growth ODE:
//!     dy/dt = y          =>  y(t) = e^t
//!
//! while simultaneously computing the integral:
//!     Q(t) = integral of y(t) dt = e^t - 1
//!
//! by augmenting the state vector to [y, Q] and adding the quadrature equation:
//!     dQ/dt = y
//!
//! This demonstrates:
//! - Encoding a quadrature as an extra ODE component
//! - Per-component tolerances via `Tolerance::Vector` to control whether the
//!   quadrature influences step-size selection
//! - Extracting quadrature results from the combined state vector

use differential_equations::ivp::Ivp;
use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use quill::prelude::*;

struct ExponentialGrowthWithQuadrature;

impl ODE<f64, SVector<f64, 2>> for ExponentialGrowthWithQuadrature {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[0];
        dydt[1] = y[0];
    }
}

fn main() {
    let ode = ExponentialGrowthWithQuadrature;
    let t0 = 0.0;
    let tf = 5.0;
    let y0: SVector<f64, 2> = vector![1.0, 0.0];

    let solution = Ivp::ode(&ode, t0, tf, y0)
        .even(0.5)
        .method(
            ExplicitRungeKutta::dop853()
                .rtol([1e-12, 1e-12])
                .atol([1e-12, 1e-12]),
        )
        .solve()
        .expect("Solver failed");

    let y_final = solution.y.last().unwrap();

    let analytical_y = tf.exp();
    let analytical_q = tf.exp() - 1.0;
    let y_error = (y_final[0] - analytical_y).abs();
    let q_error = (y_final[1] - analytical_q).abs();

    println!("Quadrature via State Augmentation");
    println!("==================================");
    println!("t\t\ty(t)\t\tQ(t)\t\ty_analytical\tQ_analytical");
    for (t, y) in solution.iter() {
        let analytical_y = t.exp();
        let analytical_q = t.exp() - 1.0;
        println!(
            "{:.4}\t\t{:.8}\t{:.8}\t{:.8}\t{:.8}",
            t, y[0], y[1], analytical_y, analytical_q
        );
    }
    println!("==================================");
    println!("y error:  {:.2e}", y_error);
    println!("Q error:  {:.2e}", q_error);

    let y_series: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[0])).collect();
    let q_series: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[1])).collect();
    let y_analytical: Vec<(f64, f64)> = (0..=50)
        .map(|i| {
            let t = i as f64 * tf / 50.0;
            (t, t.exp())
        })
        .collect();
    let q_analytical: Vec<(f64, f64)> = (0..=50)
        .map(|i| {
            let t = i as f64 * tf / 50.0;
            (t, t.exp() - 1.0)
        })
        .collect();

    Plot::builder()
        .title("Quadrature via State Augmentation: y(t) = e^t, Q(t) = integral of e^t dt")
        .x_label("t")
        .y_label("Value")
        .legend(Legend::TopLeftInside)
        .data([
            Series::builder()
                .name("y(t) = e^t")
                .color("Blue")
                .data(y_series)
                .marker(Marker::Circle)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Q(t) = integral of e^t dt")
                .color("Green")
                .data(q_series)
                .marker(Marker::Square)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("y analytical")
                .color("LightBlue")
                .data(y_analytical)
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
            Series::builder()
                .name("Q analytical")
                .color("LightGreen")
                .data(q_analytical)
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/ode/16_quadrature/quadrature.svg")
        .expect("Failed to save plot as SVG");
}
