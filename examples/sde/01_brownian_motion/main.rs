//! Example 01: Brownian Motion
//! 
//! This example simulates standard Brownian motion using the SDE:
//! dX = σ·dW
//! 
//! where:
//! - σ (sigma) is the volatility parameter (constant)
//! - dW is the increment of a standard Wiener process
//!
//! Brownian motion is a fundamental stochastic process where a particle
//! moves randomly due to collisions with molecules. It has many applications
//! in science, finance, and mathematics.

use differential_equations::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Struct representing Brownian motion with volatility parameter σ
struct BrownianMotion {
    sigma: f64,             // Volatility parameter
    rng: rand::rngs::StdRng, // Random number generator
}

impl BrownianMotion {
    /// Create a new BrownianMotion with specified volatility and random seed
    fn new(sigma: f64, seed: u64) -> Self {
        Self {
            sigma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

/// Implementation of the SDE trait for Brownian motion
impl SDE for BrownianMotion {
    /// Drift term for Brownian motion (0 for standard Brownian motion)
    fn drift(&self, _t: f64, _y: &f64, dydt: &mut f64) {
        *dydt = 0.0; // No drift for standard Brownian motion
    }

    /// Diffusion term for Brownian motion (σ)
    fn diffusion(&self, _t: f64, _y: &f64, dydw: &mut f64) {
        *dydw = self.sigma;
    }
    
    /// Generate noise for Brownian motion
    fn noise(&self, dt: f64, dw: &mut f64) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        *dw = normal.sample(&mut self.rng.clone());
    }
}

fn main() {
    // Parameters
    let t0 = 0.0;
    let tf = 5.0;
    let dt = 0.01;
    let y0 = 0.0; // Initial position at origin
    let sigma = 0.5; // Volatility parameter
    let seed = 42; // Seed for reproducibility

    // Create SDE system with seed for reproducibility
    let sde = BrownianMotion::new(sigma, seed);

    // Create solver
    let mut solver = EM::new(dt);

    println!("Simulating Brownian motion with σ = {}", sigma);
    println!("Time interval: [{}, {}], Step size: {}", t0, tf, dt);
    println!("Initial position: {}", y0);
    println!("Random seed: {}", seed);

    // Create and solve the problem
    let problem = SDEProblem::new(sde, t0, tf, y0);
    let solution = problem.solve(&mut solver).unwrap();

    // Print solution statistics
    println!("Simulation completed:");
    println!("  Number of time steps: {}", solution.t.len());
    println!("  Final position: {}", solution.y.last().unwrap());
    println!("  Function evaluations: {}", solution.evals);
    println!("  Total steps: {}", solution.steps);
    println!("  Solution time: {:.6} seconds", solution.timer.elapsed());
}