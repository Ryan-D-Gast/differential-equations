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

/// Struct representing Ornstein-Uhlenbeck process
struct OrnsteinUhlenbeck {
    theta: f64, // Mean reversion speed
    mu: f64,    // Long-term mean
    sigma: f64, // Volatility
    rng: rand::rngs::StdRng,
}

impl OrnsteinUhlenbeck {
    /// Create a new Ornstein-Uhlenbeck process with specified parameters and seed
    fn new(theta: f64, mu: f64, sigma: f64, seed: u64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

/// Implementation of the SDE trait for Ornstein-Uhlenbeck process
impl SDE for OrnsteinUhlenbeck {
    /// Mean-reverting drift term: θ(μ-X)
    fn drift(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.theta * (self.mu - *y);
    }

    /// Constant volatility: σ
    fn diffusion(&self, _t: f64, _y: &f64, dydw: &mut f64) {
        *dydw = self.sigma;
    }

    /// Generate noise for the process
    fn noise(&self, dt: f64, dw: &mut f64) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        *dw = normal.sample(&mut self.rng.clone());
    }
}

fn main() {
    // Parameters
    let t0 = 0.0;
    let tf = 10.0;
    let dt = 0.01;
    let y0 = 5.0; // Initial value, far from mean
    let theta = 0.5; // Mean reversion speed
    let mu = 1.0; // Long-term mean
    let sigma = 0.3; // Volatility parameter
    let seed = 42; // Seed for reproducibility

    println!("Simulating Ornstein-Uhlenbeck process with parameters:");
    println!("θ = {}, μ = {}, σ = {}", theta, mu, sigma);
    println!("Time interval: [{}, {}], Step size: {}", t0, tf, dt);
    println!("Initial value: {}", y0);
    println!("Random seed: {}", seed);

    // Create SDE system
    let sde = OrnsteinUhlenbeck::new(theta, mu, sigma, seed);

    // Compare both solvers
    println!("\nComparing solvers:");

    // Solve with Runge-Kutta-Maruyama
    let mut rk_solver = RKM4::new(dt);
    let rk_problem = SDEProblem::new(sde, t0, tf, y0);
    let rk_solution = rk_problem.even(1.0).solve(&mut rk_solver).unwrap();
    println!("\nRunge-Kutta-Maruyama results:");
    for (t, y) in rk_solution.iter() {
        println!("  t = {:.2}, y = {:.6}", t, y);
    }
    println!("  Function evaluations: {}", rk_solution.evals);
    println!(
        "  Solution time: {:.6} seconds",
        rk_solution.timer.elapsed()
    );

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
}
