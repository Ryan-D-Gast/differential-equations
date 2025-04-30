//! Example 02: Heston Stochastic Volatility Model
//! 
//! This example simulates the Heston stochastic volatility model using the system of SDEs:
//! dS = μS dt + √v·S dW₁
//! dv = κ(θ-v) dt + σ√v dW₂
//! 
//! where:
//! - S is the asset price
//! - v is the volatility (variance) of the asset
//! - μ is the drift of the asset price
//! - κ is the mean reversion speed of volatility
//! - θ is the long-term mean of volatility
//! - σ is the volatility of volatility
//! - dW₁, dW₂ are Wiener processes with correlation ρ
//!
//! The Heston model is widely used in finance to model assets with stochastic volatility,
//! providing more realistic price dynamics than models with constant volatility.

use differential_equations::{
    prelude::*,
    derive::State,
};
use rand::SeedableRng;
use rand_distr::Distribution;

// Custom state type for our 2D state vector (price and variance)
#[derive(State)]
struct HestonState<T> {
    price: T,      // Asset price S
    variance: T,   // Variance (volatility squared) v
}

/// Struct representing Heston stochastic volatility model
#[derive(Clone)]
struct HestonModel {
    mu: f64,      // Drift of asset price
    kappa: f64,   // Mean reversion speed of volatility
    theta: f64,   // Long-term mean of volatility
    sigma: f64,   // Volatility of volatility
    rho: f64,     // Correlation between price and volatility Wiener processes
    rng: rand::rngs::StdRng, // Random number generator
}

impl HestonModel {
    /// Create a new Heston model with the specified parameters and random seed
    fn new(mu: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, seed: u64) -> Self {
        Self {
            mu,
            kappa,
            theta,
            sigma,
            rho,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

/// Implementation of the SDE trait for Heston model
impl SDE<f64, HestonState<f64>> for HestonModel {
    fn drift(&self, _t: f64, y: &HestonState<f64>, dydt: &mut HestonState<f64>) {
        // Asset price drift: μS
        dydt.price = self.mu * y.price;
        
        // Variance drift: κ(θ-v)
        dydt.variance = self.kappa * (self.theta - y.variance);
    }

    fn diffusion(&self, _t: f64, y: &HestonState<f64>, dydw: &mut HestonState<f64>) {
        // Asset price diffusion: √v·S
        dydw.price = y.price * y.variance.sqrt();
        
        // Variance diffusion: σ√v
        dydw.variance = self.sigma * y.variance.sqrt();
    }

    fn noise(&self, dt: f64, dw_vec: &mut HestonState<f64>) {
        // Generate correlated Wiener process increments for price and variance
        let normal = rand_distr::Normal::new(0.0, dt.sqrt()).unwrap();
        
        // Generate uncorrelated increments
        let dw1 = normal.sample(&mut self.rng.clone()); // dW₁
        let dw2 = normal.sample(&mut self.rng.clone()); // dW₂
        
        // Apply correlation using Cholesky decomposition
        dw_vec.price = dw1;
        dw_vec.variance = self.rho * dw1 + (1.0 - self.rho * self.rho).sqrt() * dw2;
    }
}

fn main() {
    // Heston model parameters
    let mu = 0.1;       // Asset drift (10% annual return)
    let kappa = 2.0;    // Mean reversion speed
    let theta = 0.04;   // Long-term variance (20% annual volatility)
    let sigma = 0.3;    // Volatility of volatility
    let rho = -0.7;     // Correlation (typically negative for equity markets)
    let seed = 42;      // Seed for reproducibility
    
    // Simulation parameters
    let t0 = 0.0;
    let tf = 1.0;       // 1 year
    let dt = 0.01;      // Step size
    
    // Initial conditions
    let s0 = 100.0;     // Initial price
    let v0 = 0.04;      // Initial variance (20% volatility)
    let y0 = HestonState { price: s0, variance: v0 };
    
    println!("Simulating Heston model with parameters:");
    println!("μ = {}, κ = {}, θ = {}, σ = {}, ρ = {}", mu, kappa, theta, sigma, rho);
    println!("Time interval: [{}, {}], Step size: {}", t0, tf, dt);
    println!("Initial price: {}, Initial variance: {}", s0, v0);
    println!("Random seed: {}", seed);
    
    // Create SDE system with seed for reproducibility
    let sde = HestonModel::new(mu, kappa, theta, sigma, rho, seed);
    
    // Create solver with fixed step size
    let mut solver = Milstein::new(dt);
    
    // Create and solve the problem
    let problem = SDEProblem::new(sde, t0, tf, y0);
    let solution = problem.solve(&mut solver).unwrap();
    
    // Get final state
    let final_price = solution.y.last().unwrap().price;
    let final_variance = solution.y.last().unwrap().variance;
    let final_volatility = final_variance.sqrt();
    
    // Print solution statistics
    println!("Simulation completed:");
    println!("  Number of time steps: {}", solution.t.len());
    println!("  Final price: {:.4}", final_price);
    println!("  Final variance: {:.4}", final_variance);
    println!("  Final volatility: {:.4}%", final_volatility * 100.0);
    println!("  Function evaluations: {}", solution.evals);
    println!("  Total steps: {}", solution.steps);
    println!("  Solution time: {:.6} seconds", solution.timer.elapsed());
    
    // Print some intermediate values (e.g., every quarter)
    println!("\nIntermediate values:");
    let intervals = [0.25, 0.5, 0.75];
    for &time in &intervals {
        // Find closest time point
        let idx = solution.t.iter().position(|&t| t >= time).unwrap_or(0);
        let state = &solution.y[idx];
        println!("  t = {:.2}: price = {:.4}, volatility = {:.2}%", 
                solution.t[idx], state.price, state.variance.sqrt() * 100.0);
    }
}
