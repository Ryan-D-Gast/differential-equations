//! Example 03: Ornstein-Uhlenbeck Process
//!
//! This example simulates the Ornstein-Uhlenbeck process using the SDE:
//! dX = θ(μ-X)dt + σdW
//!
//! where:
//! - θ (theta) is the mean reversion speed
//! - μ (mu) is the long-term mean
//! - σ (sigma) is the volatility parameter
//! - dW is the increment of a standard Wiener process
//!
//! The Ornstein-Uhlenbeck process is a mean-reverting stochastic process,
//! commonly used to model systems that tend to drift toward an equilibrium point.
//! It has applications in physics, finance (for modeling interest rates), and
//! various other fields.

use differential_equations::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

struct OrnsteinUhlenbeck {
    theta: f64,
    mu: f64,
    sigma: f64,
    rng: rand::rngs::StdRng,
}

impl OrnsteinUhlenbeck {
    fn new(theta: f64, mu: f64, sigma: f64, seed: u64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

impl SDE for OrnsteinUhlenbeck {
    fn drift(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.theta * (self.mu - *y);
    }

    fn diffusion(&self, _t: f64, _y: &f64, dydw: &mut f64) {
        *dydw = self.sigma;
    }

    fn noise(&mut self, dt: f64, dw: &mut f64) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        *dw = normal.sample(&mut self.rng);
    }
}

fn main() {
    // --- Problem Configuration ---

    // Ornstein-Uhlenbeck process parameters
    let theta = 0.5; // Mean reversion speed
    let mu = 1.0; // Long-term mean
    let sigma = 0.3; // Volatility parameter
    let seed = 42; // Seed for reproducibility

    // Time settings
    let t0 = 0.0;
    let tf = 10.0;
    let y0 = 5.0; // Initial value, far from mean

    // Create the Ornstein-Uhlenbeck SDE problem
    let sde = OrnsteinUhlenbeck::new(theta, mu, sigma, seed);
    let mut problem = SDEProblem::new(sde, t0, tf, y0);

    // --- Solve the SDE ---
    let dt = 0.01;
    let mut solver = ExplicitRungeKutta::rk4(dt);
    let points_of_interest = [2.0, 5.0, 8.0];
    let solution = problem.t_eval(points_of_interest).solve(&mut solver).unwrap();

    // --- Print the results ---
    let final_value = *solution.y.last().unwrap();

    println!("Simulating Ornstein-Uhlenbeck process with parameters:");
    println!("θ = {}, μ = {}, σ = {}", theta, mu, sigma);
    println!("Time interval: [{}, {}], Step size: {}", t0, tf, dt);
    println!("Initial value: {}", y0);
    println!("Random seed: {}", seed);
    println!("Simulation completed:");
    println!("  Number of time steps: {}", solution.t.len());
    println!("  Final value: {:.6}", final_value);
    println!("  Function evaluations: {}", solution.evals.function);
    println!("  Total steps: {}", solution.steps.total());
    println!("  Solution time: {:.6} seconds", solution.timer.elapsed());

    // Expected mean and variance (analytical solution for long-time behavior)
    // For OU process, mean → μ and variance → σ²/(2θ) as t → ∞
    let expected_mean = mu;
    let expected_variance = sigma * sigma / (2.0 * theta);
    println!("\nAnalytical long-time statistics:");
    println!("  Expected mean: {}", expected_mean);
    println!(
        "  Expected standard deviation: {:.6}",
        expected_variance.sqrt()
    );

    println!("\nIntermediate values:");
    for (t, y) in solution.iter() {
        println!("  t = {:.2}: y = {:.6}", t, y);
    }
}
