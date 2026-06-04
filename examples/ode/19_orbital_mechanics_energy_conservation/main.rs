//! Example 19: Orbital Mechanics Energy Conservation
//!
//! This example simulates the Kepler problem (a 2-body orbital mechanics problem)
//! using symplectic integrators (Velocity Verlet and Ruth-Forest) to demonstrate
//! long-term energy conservation compared to a standard Runge-Kutta method.
//!
//! The Hamiltonian is:
//! H(q, p) = 1/2 * (p_x^2 + p_y^2) - 1 / sqrt(q_x^2 + q_y^2)
//!
//! where q = (q_x, q_y) is position and p = (p_x, p_y) is momentum.

use differential_equations::methods::SymplecticIntegrator;
use differential_equations::prelude::*;
use quill::prelude::*;

struct KeplerProblem;

impl Hamiltonian<f64, Vec<f64>> for KeplerProblem {
    // velocity is dq/dt = dH/dp = p
    fn velocity(&self, _t: f64, _q: &Vec<f64>, p: &Vec<f64>, dq: &mut Vec<f64>) {
        dq[0] = p[0];
        dq[1] = p[1];
    }

    // force is dp/dt = -dH/dq = -q / r^3
    fn force(&self, _t: f64, q: &Vec<f64>, _p: &Vec<f64>, dp: &mut Vec<f64>) {
        let q_x = q[0];
        let q_y = q[1];
        let r = (q_x * q_x + q_y * q_y).sqrt();
        let r3 = r * r * r;

        dp[0] = -q_x / r3;
        dp[1] = -q_y / r3;
    }
}

fn compute_energy(y: &[f64]) -> f64 {
    let q_x = y[0];
    let q_y = y[1];
    let p_x = y[2];
    let p_y = y[3];

    let r = (q_x * q_x + q_y * q_y).sqrt();
    let kinetic = 0.5 * (p_x * p_x + p_y * p_y);
    let potential = -1.0 / r;

    kinetic + potential
}

fn main() {
    let eccentricity = 0.6;
    let t0 = 0.0;
    // Many orbits (approx 10 orbits, T = 2*pi)
    let tf = 100.0 * 2.0 * std::f64::consts::PI;
    let dt = 0.05;

    // Initial conditions: starting at pericenter
    // q_x = 1 - e, q_y = 0
    // p_x = 0,     p_y = sqrt((1+e)/(1-e))
    let q_x0 = 1.0 - eccentricity;
    let p_y0 = ((1.0_f64 + eccentricity) / (1.0_f64 - eccentricity)).sqrt();
    let y0 = vec![q_x0, 0.0, 0.0, p_y0];

    let initial_energy = compute_energy(&y0);

    // 1. Runge-Kutta 4 (RK4)
    let sol_rk4 = IVP::hamiltonian(&KeplerProblem, t0, tf, y0.clone())
        .method(ExplicitRungeKutta::rk4(dt).max_steps(50000))
        .solve()
        .expect("RK4 Failed");

    // 2. Velocity Verlet (Symplectic 2nd Order)
    let sol_vv = IVP::hamiltonian(&KeplerProblem, t0, tf, y0.clone())
        .method(SymplecticIntegrator::velocity_verlet(dt).max_steps(50000))
        .solve()
        .expect("Velocity Verlet Failed");

    // 3. Ruth-Forest (Symplectic 4th Order)
    let sol_rf = IVP::hamiltonian(&KeplerProblem, t0, tf, y0.clone())
        .method(SymplecticIntegrator::ruth_forest(dt).max_steps(50000))
        .solve()
        .expect("Ruth-Forest Failed");

    // Compute energy errors
    let energy_err_rk4: Vec<_> = sol_rk4
        .iter()
        .map(|(t, y)| (*t, (compute_energy(y) - initial_energy).abs()))
        .collect();
    let energy_err_vv: Vec<_> = sol_vv
        .iter()
        .map(|(t, y)| (*t, (compute_energy(y) - initial_energy).abs()))
        .collect();
    let energy_err_rf: Vec<_> = sol_rf
        .iter()
        .map(|(t, y)| (*t, (compute_energy(y) - initial_energy).abs()))
        .collect();

    // Plot Energy Error
    Plot::builder()
        .title("Energy Conservation in Kepler Problem")
        .x_label("Time (t)")
        .y_label("Energy Error |E(t) - E(0)|")
        .legend(Legend::TopLeftInside)
        .data([
            Series::builder()
                .name("RK4 (Non-Symplectic)")
                .color("Red")
                .data(energy_err_rk4)
                .build(),
            Series::builder()
                .name("Velocity Verlet (Symplectic 2nd Order)")
                .color("Blue")
                .data(energy_err_vv)
                .build(),
            Series::builder()
                .name("Ruth-Forest (Symplectic 4th Order)")
                .color("Green")
                .data(energy_err_rf)
                .build(),
        ])
        .build()
        .to_svg("examples/ode/19_orbital_mechanics_energy_conservation/energy_error.svg")
        .expect("Failed to save energy plot");

    // Plot Orbits
    let orbit_rf: Vec<_> = sol_rf.iter().map(|(_t, y)| (y[0], y[1])).collect();
    Plot::builder()
        .title("Ruth-Forest Orbit")
        .x_label("x")
        .y_label("y")
        .data([Series::builder()
            .name("Orbit")
            .color("Green")
            .data(orbit_rf)
            .build()])
        .build()
        .to_svg("examples/ode/19_orbital_mechanics_energy_conservation/orbit.svg")
        .expect("Failed to save orbit plot");

    println!("Simulation finished.");
    println!(
        "RK4 final energy error: {:.2e}",
        (compute_energy(sol_rk4.last().unwrap().1) - initial_energy).abs()
    );
    println!(
        "Velocity Verlet final energy error: {:.2e}",
        (compute_energy(sol_vv.last().unwrap().1) - initial_energy).abs()
    );
    println!(
        "Ruth-Forest final energy error: {:.2e}",
        (compute_energy(sol_rf.last().unwrap().1) - initial_energy).abs()
    );
}
