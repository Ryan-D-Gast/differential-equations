//! Example 18: Pipe Heat Transfer Boundary Value Problem
//!
//! This example solves a steady-state heat-transfer model along a pipe or rod:
//! T'' = beta^2 * (T - T_ambient)
//!
//! where:
//! - T is the pipe temperature
//! - x is position along the pipe, not time
//! - beta controls heat loss to the surrounding ambient environment
//!
//! The boundary conditions are:
//! - fixed inlet temperature: T(0) = T_inlet
//! - insulated outlet: dT/dx(L) = 0
//!
//! This is a spatial boundary value problem, not a transient heat equation. It
//! describes the temperature profile after transients have died out while the
//! inlet temperature is still maintained. The solution is compared against the
//! analytical insulated-end profile.
//!
//! This example showcases:
//! - Defining an ODE BVP with the regular `ODE` trait plus `Boundary`
//! - Solving with `BVP::ode` and `Shooting::single`
//! - Using evenly spaced BVP output for plotting
//! - Validating the shooting result against an analytical solution

use differential_equations::prelude::*;
use quill::prelude::*;

struct PipeHeatTransfer {
    ambient_temperature: f64,
    heat_loss_rate: f64,
    inlet_temperature: f64,
}

impl PipeHeatTransfer {
    fn analytical_initial_gradient(&self, length: f64) -> f64 {
        let theta_0 = self.inlet_temperature - self.ambient_temperature;
        let beta_l = self.heat_loss_rate * length;
        -self.heat_loss_rate * theta_0 * beta_l.tanh()
    }

    fn analytical_outlet_temperature(&self, length: f64) -> f64 {
        self.analytical_temperature(length, length)
    }

    fn analytical_temperature(&self, x: f64, length: f64) -> f64 {
        let theta_0 = self.inlet_temperature - self.ambient_temperature;
        let beta_l = self.heat_loss_rate * length;
        let beta_distance_from_outlet = self.heat_loss_rate * (length - x);
        self.ambient_temperature + theta_0 * beta_distance_from_outlet.cosh() / beta_l.cosh()
    }
}

impl ODE<f64, [f64; 2]> for PipeHeatTransfer {
    fn diff(&self, _x: f64, y: &[f64; 2], dydx: &mut [f64; 2]) {
        dydx[0] = y[1];
        dydx[1] = self.heat_loss_rate.powi(2) * (y[0] - self.ambient_temperature);
    }
}

impl Boundary<f64, [f64; 2]> for PipeHeatTransfer {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        res[0] = y_a[0] - self.inlet_temperature;
        res[1] = y_b[1];
    }
}

fn main() {
    let pipe = PipeHeatTransfer {
        ambient_temperature: 293.15,
        heat_loss_rate: 0.2,
        inlet_temperature: 373.15,
    };

    let length = 10.0;
    let x0 = 0.0;
    let initial_guess = [pipe.inlet_temperature, -5.0];

    println!("Solving steady pipe heat-transfer BVP");
    println!(
        "T'' = beta^2 (T - T_ambient), T(0) = {:.2} K, dT/dx({:.1}) = 0",
        pipe.inlet_temperature, length
    );

    let ode_solver = ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12);
    let solver = Shooting::single(ode_solver)
        .tolerance(1e-8)
        .max_iterations(50);

    let result = BVP::ode(&pipe, x0, length, initial_guess)
        .even(0.25)
        .method(solver)
        .solve()
        .expect("pipe heat-transfer BVP should solve");

    let (x_l, y_l) = result.last().expect("solution should include final point");
    let (_, y_0) = result
        .iter()
        .next()
        .expect("solution should include initial point");
    let expected_gradient = pipe.analytical_initial_gradient(length);
    let expected_outlet_temperature = pipe.analytical_outlet_temperature(length);

    println!("Final x: {:.3} m, final state: {:?}", x_l, y_l);
    println!("Initial state found: {:?}", y_0);
    println!("Analytical initial gradient: {:.6} K/m", expected_gradient);
    println!(
        "Analytical outlet temperature: {:.6} K",
        expected_outlet_temperature
    );

    assert!(
        (y_0[0] - pipe.inlet_temperature).abs() < 1e-8,
        "inlet boundary temperature should be satisfied"
    );
    assert!(
        y_l[1].abs() < 1e-5,
        "insulated outlet should have zero temperature gradient"
    );
    assert!(
        (y_0[1] - expected_gradient).abs() < 1e-4,
        "shooting method should recover the analytical initial temperature gradient"
    );
    assert!(
        (y_l[0] - expected_outlet_temperature).abs() < 1e-4,
        "outlet temperature should match the analytical insulated-end solution"
    );

    Plot::builder()
        .title("Steady Pipe Heat-Transfer BVP")
        .x_label("Position x (m)")
        .y_label("Temperature (K)")
        .legend(Legend::TopRightInside)
        .data([
            Series::builder()
                .name("Numerical BVP solution")
                .color("Blue")
                .data(result.iter().map(|(x, y)| (*x, y[0])).collect::<Vec<_>>())
                .marker(Marker::Circle)
                .marker_size(3.0)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Analytical insulated-end profile")
                .color("Red")
                .data(
                    (0..=100)
                        .map(|i| {
                            let x = length * i as f64 / 100.0;
                            (x, pipe.analytical_temperature(x, length))
                        })
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/ode/18_pipe_heat_transfer/pipe_heat_transfer.svg")
        .expect("Failed to save plot as SVG");

    println!("Pipe heat-transfer BVP solved successfully!");
}
