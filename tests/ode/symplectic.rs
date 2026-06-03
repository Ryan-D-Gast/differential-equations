use differential_equations::interpolate::Interpolation;
use differential_equations::methods::SymplecticIntegrator;
use differential_equations::ode::{Hamiltonian, HamiltonianSystem, ODE, OrdinaryNumericalMethod};
use differential_equations::prelude::*;

struct SimpleHarmonicOscillator;

// Keep the ODE impl to verify backwards compatibility and raw ODE execution
impl ODE<f64, Vec<f64>> for SimpleHarmonicOscillator {
    // y = [q, p]
    fn diff(&self, _t: f64, y: &Vec<f64>, dydt: &mut Vec<f64>) {
        dydt[0] = y[1]; // dq/dt = p
        dydt[1] = -y[0]; // dp/dt = -q
    }
}

// Implement the new safe Hamiltonian trait
impl Hamiltonian<f64> for SimpleHarmonicOscillator {
    fn velocity(&self, _t: f64, _q: &[f64], p: &[f64], dq: &mut [f64]) {
        dq.copy_from_slice(p);
    }

    fn force(&self, _t: f64, q: &[f64], _p: &[f64], dp: &mut [f64]) {
        for i in 0..q.len() {
            dp[i] = -q[i];
        }
    }
}

#[test]
fn test_symplectic_odd_dimension_returns_error() {
    let system = SimpleHarmonicOscillator;

    // Odd length state
    let y0_odd = vec![1.0, 0.0, 0.0];
    let result_vv = IVP::hamiltonian(&system, 0.0, 1.0, y0_odd.clone())
        .method(SymplecticIntegrator::velocity_verlet(0.1))
        .solve();

    assert!(result_vv.is_err());
    if let Err(Error::BadInput { msg }) = result_vv {
        assert!(msg.contains("even number of elements"));
    } else {
        panic!("Expected Error::BadInput");
    }

    let result_rf = IVP::hamiltonian(&system, 0.0, 1.0, y0_odd)
        .method(SymplecticIntegrator::ruth_forest(0.1))
        .solve();

    assert!(result_rf.is_err());
}

#[test]
fn test_symplectic_empty_dimension_returns_error() {
    let system = SimpleHarmonicOscillator;
    let y0_empty = vec![];
    let result = IVP::hamiltonian(&system, 0.0, 1.0, y0_empty)
        .method(SymplecticIntegrator::velocity_verlet(0.1))
        .solve();
    assert!(result.is_err());
}

#[test]
fn test_symplectic_solves_harmonic_oscillator_trait() {
    let system = SimpleHarmonicOscillator;
    let y0 = vec![1.0, 0.0]; // q=1, p=0

    let solution_vv = IVP::hamiltonian(&system, 0.0, 2.0 * std::f64::consts::PI, y0.clone())
        .method(SymplecticIntegrator::velocity_verlet(0.01))
        .solve()
        .expect("Velocity Verlet should solve with Hamiltonian trait");

    let final_y_vv = solution_vv.y.last().unwrap();
    // After one full period (2*pi), it should return close to initial state
    assert!((final_y_vv[0] - 1.0).abs() < 1.0e-3);
    assert!((final_y_vv[1] - 0.0).abs() < 1.0e-3);

    let solution_rf = IVP::hamiltonian(&system, 0.0, 2.0 * std::f64::consts::PI, y0)
        .method(SymplecticIntegrator::ruth_forest(0.01))
        .solve()
        .expect("Ruth-Forest should solve with Hamiltonian trait");

    let final_y_rf = solution_rf.y.last().unwrap();
    assert!((final_y_rf[0] - 1.0).abs() < 1.0e-5);
    assert!((final_y_rf[1] - 0.0).abs() < 1.0e-5);
}

#[test]
fn test_symplectic_solves_harmonic_oscillator_from_fn() {
    let y0 = vec![1.0, 0.0]; // q=1, p=0

    let velocity = |_t: f64, _q: &[f64], p: &[f64], dq: &mut [f64]| {
        dq.copy_from_slice(p);
    };
    let force = |_t: f64, q: &[f64], _p: &[f64], dp: &mut [f64]| {
        dp[0] = -q[0];
    };

    let solution_vv =
        IVP::hamiltonian_from_fn(velocity, force, 0.0, 2.0 * std::f64::consts::PI, y0)
            .method(SymplecticIntegrator::velocity_verlet(0.01))
            .solve()
            .expect("Velocity Verlet should solve with hamiltonian_from_fn");

    let final_y_vv = solution_vv.y.last().unwrap();
    assert!((final_y_vv[0] - 1.0).abs() < 1.0e-3);
    assert!((final_y_vv[1] - 0.0).abs() < 1.0e-3);
}

#[test]
fn test_symplectic_linear_interpolation() {
    let system = SimpleHarmonicOscillator;
    let ode_adapter = HamiltonianSystem::new(&system);
    let y0 = vec![1.0, 0.0];

    let mut solver = SymplecticIntegrator::velocity_verlet(0.1);
    solver.init(&ode_adapter, 0.0, 1.0, &y0).unwrap();
    solver.step(&ode_adapter).unwrap();

    // t should be 0.1, t_prev should be 0.0
    assert_eq!(solver.t(), 0.1);
    assert_eq!(solver.t_prev(), 0.0);

    let y_prev = solver.y_prev().clone();
    let y_curr = solver.y().clone();

    // Interpolate halfway
    let y_half: Vec<f64> = solver.interpolate(0.05).unwrap();
    assert_eq!(y_half.len(), 2);
    assert_eq!(y_half[0], 0.5 * (y_prev[0] + y_curr[0]));
    assert_eq!(y_half[1], 0.5 * (y_prev[1] + y_curr[1]));

    // Out of bounds interpolation should fail
    assert!(solver.interpolate(-0.01).is_err());
    assert!(solver.interpolate(0.11).is_err());
}
