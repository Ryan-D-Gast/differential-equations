# Stochastic Differential Equations (SDE)

The `sde` module provides tools for solving stochastic differential equations (SDEs), which are differential equations with both deterministic and random components.

## Table of Contents

- [Defining an SDE](#defining-an-sde)
- [Solving an SDE Problem](#solving-an-sde-problem)
- [Examples](#examples)
- [Notation](#notation)

## Defining an SDE

The `SDE` trait defines a stochastic differential equation of the form dY = a(t,Y)dt + b(t,Y)dW for the solver, where:
- a(t,Y) is the drift function (deterministic part)
- b(t,Y) is the diffusion function (stochastic part)
- dW is the increment of a Wiener process

The trait also includes a `noise` method to generate random increments and an optional `event` function to interrupt the solver when a specific condition is met.

### SDE Trait
* `drift` - Deterministic part a(t,Y) of the SDE in form `drift(t, &y, &mut dydt)`.
* `diffusion` - Stochastic part b(t,Y) of the SDE in form `diffusion(t, &y, &mut dydw)`.
* `noise` - Generates random noise increments for the SDE.

### Event Trait
For event detection with precise zero-crossing detection, implement the separate `Event` trait:

* `config` - Configure event detection parameters (direction filtering, termination count)
* `event` - Event function `g(t,y)` whose zero crossings are detected using Brent-Dekker root finding

### Implementation
```rust
use differential_equations::prelude::*;
use differential_equations::solout::{Event, EventConfig, CrossingDirection};
use nalgebra::SVector;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// Geometric Brownian Motion
struct GBM {
    mu: f64,     // Drift parameter
    sigma: f64,  // Volatility parameter
    rng: rand::rngs::StdRng, // Random number generator
}

impl GBM {
    fn new(mu: f64, sigma: f64, seed: u64) -> Self {
        Self {
            mu,
            sigma,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

// Implement the SDE trait for GBM
impl SDE<f64, SVector<f64, 1>> for GBM {
    // Drift term: μY
    fn drift(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.mu * y[0];
    }
    
    // Diffusion term: σY
    fn diffusion(&self, _t: f64, y: &SVector<f64, 1>, dydw: &mut SVector<f64, 1>) {
        dydw[0] = self.sigma * y[0];
    }
    
    // Generate random noise increments
    fn noise(&self, dt: f64, dw: &mut SVector<f64, 1>) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        dw[0] = normal.sample(&mut self.rng.clone());
    }
}

// Separate event detection implementation
impl Event<f64, SVector<f64, 1>> for GBM {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Positive, Some(1)) // Terminate after first event
    }

    fn event(&self, _t: f64, y: &SVector<f64, 1>) -> f64 {
        // Event function g(t,y) = y - 150.0
        // Zero crossing occurs when y = 150.0
        y[0] - 150.0
    }
}
```

## Solving an SDE Problem

The `SDEProblem` struct is used to set up and solve an SDE. The `solve` method returns a `Result<SDESolution, Error>` where `SDESolution` contains the solution vectors and solver statistics.

```rust
fn main() {
    // Parameters
    let t0 = 0.0;
    let tf = 1.0;
    let y0 = SVector::new(100.0); // Initial stock price
    let mu = 0.1;    // 10% expected return
    let sigma = 0.3; // 30% volatility
    let seed = 42;   // For reproducibility
    
    // Create SDE system
    let sde = GBM::new(mu, sigma, seed);
    
    // Create solver with fixed step size
    let mut solver = ExplicitRungeKutta::rk4(0.01);
    
    // Create and solve the problem
    let problem = SDEProblem::new(&sde, t0, tf, y0);
    let solution = problem
        .event(&sde)  // Add event detection
        .solve(&mut solver)
        .unwrap();
    
    // Check if solver terminated due to event
    if let Status::Interrupted = solution.status {
        println!("Solver terminated due to event detection");
    }
    
    // Access solution
    println!("Final price: {:.4}", solution.y.last().unwrap()[0]);
    
    // Print some statistics
    println!("Function evaluations: {}", solution.evals.function);
    println!("Number of time steps: {}", solution.t.len());
    println!("Solution time: {:.6} seconds", solution.timer.elapsed());
}
```

## Examples

For more examples, see the `examples/sde` directory:

| Example | Description |
|---------|-------------|
| [Brownian Motion](../../examples/sde/01_brownian_motion/main.rs) | Simulates standard Brownian motion. |
| [Heston Model](../../examples/sde/02_heston_model/main.rs) | Implements the Heston stochastic volatility model for asset pricing. |
| [Ornstein-Uhlenbeck](../../examples/sde/03_ornstein_uhlenbeck/main.rs) | Simulates the Ornstein-Uhlenbeck process. |

## Notation

The SDE module uses the following notation:
- `t` - The independent variable, typically time.
- `y` - The dependent variable(s), the state of the system.
- `drift` - The deterministic part of the SDE (a(t,Y)).
- `diffusion` - The stochastic part of the SDE (b(t,Y)).
- `dw` - The increment of the Wiener process.

## Future Development

Future versions of the library will include:
- Additional SDE solvers for specific problem types
- Support for multi-dimensional correlated Wiener processes
- Specialized solvers for stiff SDEs
- Adaptive step size methods for SDEs
