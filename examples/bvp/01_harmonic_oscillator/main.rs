use differential_equations::prelude::*;
use std::f64::consts::PI;

struct HarmonicOscillatorBvp {
    omega_sq: f64,
    y_tf: f64,
}

impl ODE<f64, [f64; 2]> for HarmonicOscillatorBvp {
    fn diff(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega_sq * y[0];
    }
}

impl Boundary<f64, [f64; 2]> for HarmonicOscillatorBvp {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        // y(0) = 0
        res[0] = y_a[0] - 0.0;
        // y(pi/2) = 1.0
        res[1] = y_b[0] - self.y_tf;
    }
}

fn main() {
    let bvp = HarmonicOscillatorBvp {
        omega_sq: 1.0,
        y_tf: 1.0, // sin(pi/2) = 1.0
    };

    // Guess: y(0) = 0.0, y'(0) = 0.5
    let y_guess = [0.0, 0.5];
    let t0 = 0.0;
    let tf = PI / 2.0;

    println!("Solving BVP: y'' + y = 0, y(0) = 0, y(pi/2) = 1");

    let solver = ShootingMethod::new(ExplicitRungeKutta::dop853());

    let result = BVP::ode(&bvp, t0, tf, y_guess)
        .method(solver)
        .solve()
        .expect("Failed to solve BVP");

    let (t_f, y_f) = result.last().unwrap();
    println!("Final time: {:.6}, Final state: {:?}", t_f, y_f);

    let (_, y_first) = result.iter().next().unwrap();
    println!("Initial state found: {:?}", y_first);

    assert!(
        (y_first[1] - 1.0).abs() < 1e-5,
        "Initial velocity should be 1.0 (sin'(0) = cos(0) = 1)"
    );
    assert!(
        (y_f[0] - 1.0).abs() < 1e-5,
        "Final position should be 1.0 (sin(pi/2) = 1)"
    );

    println!("BVP solved successfully!");
}
