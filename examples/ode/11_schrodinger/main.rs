//! Example 11: Time-dependent Schrödinger Equation
//!
//! This example solves the time-dependent Schrödinger equation for a single quantum state:
//! iħ·d|ψ⟩/dt = H|ψ⟩
//!
//! where:
//! - |ψ⟩ is the quantum state (a complex-valued wavefunction)
//! - H is the Hamiltonian operator
//! - ħ is the reduced Planck constant (set to 1 for simplicity)
//! - i is the imaginary unit
//!
//! For an energy eigenstate, H|ψ⟩ = E|ψ⟩, where E is the energy eigenvalue.
//! This simplifies the equation to: d|ψ⟩/dt = -iE|ψ⟩/ħ
//!
//! The solution to this equation is |ψ(t)⟩ = |ψ(0)⟩·e^(-iEt/ħ), which has
//! constant probability amplitude but a rotating phase, demonstrating the
//! time evolution of quantum states. This is a fundamental equation in quantum
//! mechanics, governing how quantum systems evolve over time.
//!
//! This example demonstrates:
//! - Using complex numbers as state variables
//! - Conservation properties in quantum mechanical simulations
//! - Phase evolution in quantum systems
//! - Verifying numerical accuracy against analytical predictions

use differential_equations::prelude::*;
use num_complex::Complex;

/// Time-dependent Schrödinger equation for a single quantum state
/// This models a single energy eigenstate with energy E
struct ScalarSchrodingerEquation {
    energy: f64, // Energy of the quantum state
}

// Implementation for a scalar complex state
impl ODE<f64, Complex<f64>> for ScalarSchrodingerEquation {
    fn diff(&self, _t: f64, psi: &Complex<f64>, dpsi_dt: &mut Complex<f64>) {
        // Schrödinger equation: i * hbar * d/dt |psi⟩ = H |psi⟩
        // With H|psi⟩ = E|psi⟩ for an energy eigenstate:
        // d/dt |psi⟩ = -i * E |psi⟩ / hbar
        // Using hbar = 1 for simplicity: d/dt |psi⟩ = -i * E |psi⟩

        let i = Complex::new(0.0, 1.0);
        *dpsi_dt = -i * self.energy * (*psi);
    }
}

fn main() {
    // Initialize the numerical method
    let mut method = ExplicitRungeKutta::dopri5().rtol(1e-8).atol(1e-8);

    // Define the system parameters
    let energy = 1.0; // Energy eigenvalue
    let ode = ScalarSchrodingerEquation { energy };

    // Define initial condition - starting with a simple phase
    let psi0 = Complex::new(1.0, 0.0); // Initial state with amplitude 1 and phase 0

    let t0 = 0.0;
    let tf = 10.0; // Simulate for 10 time units

    // Create the ODEProblem
    let schrodinger_problem = ODEProblem::new(ode, t0, tf, psi0);

    // Solve the ODEProblem
    match schrodinger_problem.even(0.5).solve(&mut method) {
        Ok(solution) => {
            println!("Solution:");
            println!("Time, Re(ψ), Im(ψ), |ψ|²");

            for (t, psi) in solution.iter() {
                println!(
                    "{:.4}, {:.6}, {:.6}, {:.6}",
                    t,
                    psi.re,
                    psi.im,
                    psi.norm_sqr()
                );
            }

            // For an energy eigenstate, the probability |ψ|² should remain constant
            // But the phase will rotate in time according to e^(-iEt/ħ)
            println!(
                "\nExpected behavior: probability constant, phase rotating at frequency {}/ħ",
                energy
            );

            // Calculate the phase evolution between two time points
            if solution.t.len() >= 2 {
                let t1 = solution.t[0];
                let t2 = solution.t[solution.t.len() - 1];
                let psi1 = &solution.y[0];
                let psi2 = &solution.y[solution.y.len() - 1];

                let phase1 = psi1.arg();
                let phase2 = psi2.arg();
                println!("\nPhase at t={:.4}: {:.6}", t1, phase1);
                println!("Phase at t={:.4}: {:.6}", t2, phase2);
                println!("Phase change: {:.6}", phase2 - phase1);
            }

            println!("\nSimulation statistics:");
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);

            // Verify that probability is conserved (should remain at 1.0)
            let final_psi = solution.y.last().unwrap();
            println!(
                "Final probability: {:.10} (should be 1.0)",
                final_psi.norm_sqr()
            );
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
