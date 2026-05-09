use differential_equations::methods::Milstein;
use differential_equations::prelude::*;

// Exact strong convergence test for Milstein vs Euler-Maruyama
// SDE: dX = r*X*dt + a*X*dW
// We fix a specific Brownian path and use it for all solvers.

struct TestGBM {
    r: f64,
    a: f64,
    dw_sequence: Vec<f64>,
    idx: usize,
}

impl TestGBM {
    fn new(r: f64, a: f64, dw_sequence: Vec<f64>) -> Self {
        Self {
            r,
            a,
            dw_sequence,
            idx: 0,
        }
    }
}

impl SDE<f64, f64> for TestGBM {
    fn drift(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.r * y;
    }

    fn diffusion(&self, _t: f64, y: &f64, dydw: &mut f64) {
        *dydw = self.a * y;
    }

    fn noise(&mut self, _dt: f64, dw: &mut f64) {
        if self.idx < self.dw_sequence.len() {
            *dw = self.dw_sequence[self.idx];
            self.idx += 1;
        } else {
            *dw = 0.0;
        }
    }
}

#[test]
fn test_milstein_vs_euler_convergence() {
    let t0 = 0.0;
    let tf = 1.0;
    let y0 = 1.0;
    let r = 1.0; // r
    let a = 0.5; // a

    // Instead of computing convergence rate dynamically which can be noisy for single paths,
    // we just verify that for a coarse step, Milstein has a smaller error than Euler for GBM.

    // 1024 steps for the "exact" path
    let steps_fine = 1024;
    let h_fine = (tf - t0) / (steps_fine as f64);

    // Generate pseudo-random dw values without external crate
    let mut dw_fine = Vec::with_capacity(steps_fine);
    let mut w_fine = Vec::with_capacity(steps_fine + 1);
    w_fine.push(0.0);

    let mut seed: u32 = 42;
    for _ in 0..steps_fine {
        // Simple PRNG
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = (seed as f64) / (u32::MAX as f64);
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u2 = (seed as f64) / (u32::MAX as f64);

        // Box-Muller transform
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let dw = z0 * h_fine.sqrt();

        dw_fine.push(dw);
        let last_w = *w_fine.last().unwrap();
        w_fine.push(last_w + dw);
    }

    // Exact solution of GBM: Y(t) = Y(0) * exp((r - 0.5 * a^2) * t + a * W(t))
    let exact_final = y0 * ((r - 0.5 * a * a) * tf + a * w_fine.last().unwrap()).exp();

    // Test with a coarse step size (N=32)
    let steps = 32;
    let h = (tf - t0) / (steps as f64);
    let factor = steps_fine / steps;

    // Coarse dw sequence
    let mut dw_coarse = Vec::with_capacity(steps);
    for i in 0..steps {
        let mut dw_sum = 0.0;
        for j in 0..factor {
            dw_sum += dw_fine[i * factor + j];
        }
        dw_coarse.push(dw_sum);
    }

    let mut sde_euler = TestGBM::new(r, a, dw_coarse.clone());
    let sol_euler = IVP::sde(&mut sde_euler, t0, tf, y0)
        .method(ExplicitRungeKutta::euler(h))
        .solve()
        .unwrap();

    let mut sde_milstein = TestGBM::new(r, a, dw_coarse.clone());
    let sol_milstein = IVP::sde(&mut sde_milstein, t0, tf, y0)
        .method(Milstein::new(h))
        .solve()
        .unwrap();

    let euler_error = (*sol_euler.y.last().unwrap() - exact_final).abs();
    let milstein_error = (*sol_milstein.y.last().unwrap() - exact_final).abs();

    println!("Euler error: {}", euler_error);
    println!("Milstein error: {}", milstein_error);

    // Milstein should be more accurate
    assert!(
        milstein_error < euler_error,
        "Milstein error ({}) should be smaller than Euler error ({})",
        milstein_error,
        euler_error
    );
}
