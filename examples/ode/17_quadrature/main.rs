//! Example 17: Numerical Quadrature via State Augmentation
//!
//! This example demonstrates how to compute a numerical integral alongside an ODE
//! by encoding the quadrature as an additional equation in the system. No special
//! quadrature solver or callback is needed — the existing adaptive step-size control
//! and per-component tolerances handle everything.
//!
//! We model a damped mass-spring system:
//!
//!     m * x'' + c * x' + k * x = 0
//!
//! rewritten as a first-order system:
//!     dx/dt = v
//!     dv/dt = -(k/m)*x - (c/m)*v
//!
//! While solving for position and velocity, we simultaneously compute the cumulative
//! energy dissipated by the damper as a quadrature:
//!
//!     W(t) = integral of c * v(t)^2 dt
//!
//! This is physically the work done by viscous damping — the integrand c*v^2 is the
//! instantaneous power dissipated. The total energy dissipated must equal the initial
//! stored energy (0.5 * k * x0^2 when starting from rest), providing an analytical
//! check on accuracy.
//!
//! The state vector is augmented to [x, v, W]:
//!     dx/dt = v
//!     dv/dt = -(k/m)*x - (c/m)*v
//!     dW/dt = c * v^2
//!
//! This demonstrates:
//! - Encoding a physically meaningful quadrature as an extra ODE component
//! - Per-component tolerances: tight (1e-12) on position and velocity so they
//!   drive step-size selection, and loose (1e-6) on the quadrature so it does
//!   not force additional step rejections while still riding along at full accuracy
//! - Verifying quadrature accuracy against a conservation-of-energy check

use differential_equations::ivp::IVP;
use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use quill::prelude::*;

struct DampedOscillator {
    k: f64,
    m: f64,
    c: f64,
}

impl ODE<f64, SVector<f64, 3>> for DampedOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 3>, dydt: &mut SVector<f64, 3>) {
        let x = y[0];
        let v = y[1];
        dydt[0] = v;
        dydt[1] = -(self.k / self.m) * x - (self.c / self.m) * v;
        dydt[2] = self.c * v * v;
    }
}

fn main() {
    let k = 4.0;
    let m = 1.0;
    let c = 0.4;
    let ode = DampedOscillator { k, m, c };

    let t0 = 0.0;
    let tf = 20.0;
    let y0: SVector<f64, 3> = vector![1.0, 0.0, 0.0];

    let e0 = 0.5 * k * y0[0] * y0[0] + 0.5 * m * y0[1] * y0[1];

    let solution = IVP::ode(&ode, t0, tf, y0)
        .even(0.2)
        .method(
            ExplicitRungeKutta::dop853()
                .rtol([1e-12, 1e-12, 1e-6])
                .atol([1e-12, 1e-12, 1e-6]),
        )
        .solve()
        .expect("Solver failed");

    let final_state = solution.y.last().unwrap();
    let x_final = final_state[0];
    let v_final = final_state[1];
    let w_final = final_state[2];
    let e_final = 0.5 * k * x_final * x_final + 0.5 * m * v_final * v_final;

    println!("Damped Oscillator with Energy Dissipation Quadrature");
    println!("=====================================================");
    println!("Parameters: k={}, m={}, c={}", k, m, c);
    println!("Initial energy E0 = {:.10}", e0);
    println!();
    println!("t\t\tx(t)\t\tv(t)\t\tW(t)");
    for (t, y) in solution.iter() {
        println!("{:.2}\t\t{:+.8}\t{:+.8}\t{:.8}", t, y[0], y[1], y[2]);
    }
    println!("=====================================================");
    println!("Remaining mechanical energy: {:.10}", e_final);
    println!("Energy dissipated (W):      {:.10}", w_final);
    println!("Sum (should equal E0):      {:.10}", e_final + w_final);
    println!(
        "Energy conservation error:  {:.2e}",
        (e0 - e_final - w_final).abs()
    );

    let x_series: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[0])).collect();
    let w_series: Vec<(f64, f64)> = solution.iter().map(|(t, y)| (*t, y[2])).collect();
    let e_series: Vec<(f64, f64)> = solution
        .iter()
        .map(|(t, y)| {
            let ke = 0.5 * m * y[1] * y[1];
            let pe = 0.5 * k * y[0] * y[0];
            (*t, ke + pe)
        })
        .collect();

    Plot::builder()
        .title("Damped Oscillator: Position and Dissipated Energy")
        .x_label("Time (t)")
        .y_label("Value")
        .legend(Legend::TopRightInside)
        .data([
            Series::builder()
                .name("Position x(t)")
                .color("Blue")
                .data(x_series)
                .marker(Marker::Circle)
                .marker_size(2.0)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Energy dissipated W(t)")
                .color("Red")
                .data(w_series)
                .marker(Marker::Square)
                .marker_size(2.0)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Mechanical energy E(t)")
                .color("Green")
                .data(e_series)
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/ode/17_quadrature/quadrature.svg")
        .expect("Failed to save plot as SVG");
}
