//! Example 04: SDE Monte Carlo Simulation
//!
//! This example demonstrates running many stochastic differential equation simulations
//! in parallel using the `rayon` parallel solver integration.
//!
//! We simulate geometric Brownian motion (GBM), which is commonly used to model
//! stock prices in finance. The SDE is:
//! dS = μ·S·dt + σ·S·dW

use differential_equations::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

struct GeometricBrownianMotion {
    mu: f64,
    sigma: f64,
    rng: rand::rngs::StdRng,
}

impl GeometricBrownianMotion {
    fn new(mu: f64, sigma: f64, seed: u64) -> Self {
        Self {
            mu,
            sigma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

impl SDE for GeometricBrownianMotion {
    fn drift(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.mu * y;
    }

    fn diffusion(&self, _t: f64, y: &f64, dydw: &mut f64) {
        *dydw = self.sigma * y;
    }

    fn noise(&mut self, dt: f64, dw: &mut f64) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        *dw = normal.sample(&mut self.rng);
    }
}

fn main() {
    // --- Problem Configuration ---
    let t0 = 0.0;
    let tf = 1.0;
    let y0 = 100.0;
    let mu = 0.05;
    let sigma = 0.2;
    let dt = 0.01;
    let num_simulations = 1000;

    println!("Running {} SDE simulations...", num_simulations);

    let start = std::time::Instant::now();

    // Create a vector of systems (one per simulation, each with a different seed)
    // We cannot share a single SDE system because `IVP::sde` takes `&mut system`, and mutating
    // the system state (the RNG) concurrently is not thread-safe. So we create multiple.
    let mut systems = Vec::new();
    for i in 0..num_simulations {
        systems.push(GeometricBrownianMotion::new(mu, sigma, i as u64));
    }

    // Solve in parallel
    let ivps: Vec<_> = systems
        .iter_mut()
        .map(|sde| IVP::sde(sde, t0, tf, y0).method(ExplicitRungeKutta::euler(dt)))
        .collect();

    let results: Vec<_> = ivps.into_par_iter().map(|ivp| ivp.solve()).collect();
    let mut final_prices = Vec::new();

    for res in results {
        let solution = res.unwrap();
        final_prices.push(*solution.y.last().unwrap());
    }

    let mean_price: f64 = final_prices.iter().sum::<f64>() / num_simulations as f64;
    let expected_price = y0 * (mu * tf).exp();

    println!("Completed in {:?}", start.elapsed());
    println!("Monte Carlo Mean Final Price: {:.2}", mean_price);
    println!("Theoretical Expected Price:   {:.2}", expected_price);
}
