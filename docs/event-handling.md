# Event Handling

Event handling in this library has been redesigned to provide robust, SciPy-like event detection with precise zero-crossing detection using Brent-Dekker root finding. This guide explains how to implement and use the new event system effectively.

## The Event Trait

The core of the new event system is the `Event` trait:

```rust
pub trait Event<T: Real = f64, Y: State<T> = f64> {
    /// Configure the event detection parameters (called once at initialization).
    fn config(&self) -> EventConfig {
        EventConfig::default()
    }

    /// Event function g(t,y) whose zero crossings are detected.
    fn event(&self, t: T, y: &Y) -> T;
}
```

### EventConfig

The `EventConfig` struct controls how events are detected:

```rust
pub struct EventConfig {
    /// Direction of zero crossing to detect
    pub direction: CrossingDirection,
    /// Number of events before termination
    pub terminate: Option<u32>,
}
```

#### CrossingDirection

- `CrossingDirection::Both`: Detect crossings in either direction
- `CrossingDirection::Positive`: Only detect when g(t,y) crosses from negative to positive
- `CrossingDirection::Negative`: Only detect when g(t,y) crosses from positive to negative

### Basic Event Implementation

```rust
use differential_equations::prelude::*;
use differential_equations::solout::{Event, EventConfig, CrossingDirection};

struct PopulationThreshold {
    threshold: f64,
}

impl PopulationThreshold {
    fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Event for PopulationThreshold {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Positive, Some(1)) // Terminate after first event
    }

    fn event(&self, _t: f64, y: &f64) -> f64 {
        // Event function: g(t,y) = y - threshold
        // Zero crossing occurs when y = threshold
        y - self.threshold
    }
}
```

## Using Events with Problems

All problem types (ODE, SDE, DDE, DAE) support the new event system through the `.event()` method:

### ODE Problems

```rust
let event_detector = PopulationThreshold::new(9.0);
let problem = ODEProblem::new(system, t0, tf, y0);

// Add event detection to the problem
let solution = problem
    .even(1.0)        // Output every 1.0 time units
    .event(&event_detector)  // Add event detection
    .solve(&mut solver)
    .unwrap();

// Check if solver terminated due to event
if let Status::Interrupted = solution.status {
    println!("Integration terminated by event");
}
```

### SDE, DDE, and DAE Problems

The same pattern works for all problem types:

```rust
// SDE example
let sde_problem = SDEProblem::new(system, t0, tf, y0);
let solution = sde_problem.event(&event_detector).solve(&mut solver).unwrap();

// DDE example
let dde_problem = DDEProblem::new(system, t0, tf, y0, history_fn);
let solution = dde_problem.event(&event_detector).solve(&mut solver).unwrap();

// DAE example
let dae_problem = DAEProblem::new(system, t0, tf, y0);
let solution = dae_problem.event(&event_detector).solve(&mut solver).unwrap();
```

## Event Detection Algorithm

The event detection uses a robust algorithm:

1. **Sign monitoring**: Track the sign of `g(t,y)` across solver steps
2. **Zero-crossing detection**: Detect when `g(t,y)` changes sign
3. **Direction filtering**: Apply the configured `CrossingDirection` filter
4. **Root finding**: Use Brent-Dekker algorithm to locate the exact event time
5. **Point recording**: Add the precise event point to the solution
6. **Termination control**: Terminate integration if configured

### Brent-Dekker Root Finding

The library uses the Brent-Dekker algorithm for robust root finding:

- **Hybrid approach**: Combines bisection, secant, and inverse quadratic interpolation
- **Guaranteed convergence**: Always converges to a root within the bracket
- **High accuracy**: Achieves machine precision for most problems
- **Robust bracketing**: Handles edge cases and numerical instabilities

## Advanced Event Configuration

### Multiple Events

You can detect multiple events by configuring the termination count:

```rust
impl Event for MultipleThresholds {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Both, Some(5)) // Detect 5 events
    }

    fn event(&self, _t: f64, y: &f64) -> f64 {
        y - self.threshold
    }
}
```

### Continuous Monitoring

For continuous monitoring without termination:

```rust
impl Event for ContinuousMonitor {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Both, None) // No termination
    }

    fn event(&self, _t: f64, y: &f64) -> f64 {
        y - self.threshold
    }
}
```

### Direction-Specific Detection

```rust
// Only detect when crossing from below to above
let config = EventConfig::new(CrossingDirection::Positive, Some(1));

// Only detect when crossing from above to below
let config = EventConfig::new(CrossingDirection::Negative, Some(1));

// Detect crossings in both directions
let config = EventConfig::new(CrossingDirection::Both, Some(1));
```

## Combining Events with Other Solout Methods

The event system is designed to compose with other output methods:

```rust
let solution = problem
    .dense(10)        // Dense output with 10 points per step
    .event(&detector) // Plus event detection
    .solve(&mut solver)
    .unwrap();
```

This creates an `EventWrappedSolout` that:
1. Delegates to the base solout (dense output in this case)
2. Adds event detection on top
3. Records both regular output points and event points

## Custom Event Implementations

### Multi-Component Events

```rust
struct MultiComponentEvent {
    x_threshold: f64,
    y_threshold: f64,
}

impl Event<f64, Vector2<f64>> for MultiComponentEvent {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Both, Some(1))
    }

    fn event(&self, _t: f64, y: &Vector2<f64>) -> f64 {
        // Event when either component crosses its threshold
        // Using min() creates a compound event function
        (y[0] - self.x_threshold).min(y[1] - self.y_threshold)
    }
}
```

### Time-Based Events

```rust
struct PeriodicEvent {
    period: f64,
    phase: f64,
}

impl Event for PeriodicEvent {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Positive, None)
    }

    fn event(&self, t: f64, _y: &f64) -> f64 {
        // Event at regular time intervals
        (t - self.phase).sin() * self.period
    }
}
```

## Event Solout Internals

For advanced users, you can use `EventSolout` directly:

```rust
use differential_equations::solout::{EventSolout, EventWrappedSolout};

// Direct EventSolout usage
let mut event_solout = EventSolout::new(&event_detector, t0, tf);
let solution = problem.solout(&mut event_solout).solve(&mut solver).unwrap();

// EventWrappedSolout for composition
let base_solout = EvenSolout::new(1.0);
let mut wrapped = EventWrappedSolout::new(base_solout, &event_detector, t0, tf);
let solution = problem.solout(&mut wrapped).solve(&mut solver).unwrap();
```

## Examples

### Logistic Growth with Event Detection

```rust
use differential_equations::prelude::*;
use differential_equations::solout::{Event, EventConfig, CrossingDirection};

struct LogisticGrowth {
    k: f64,
    m: f64,
}

impl ODE for LogisticGrowth {
    fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.k * y * (1.0 - y / self.m);
    }
}

impl Event for LogisticGrowth {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Positive, Some(1))
    }

    fn event(&self, _t: f64, y: &f64) -> f64 {
        // Detect when population reaches 90% of carrying capacity
        y - 0.9 * self.m
    }
}

fn main() {
    let mut method = ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12);
    let y0 = 1.0;
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_problem = ODEProblem::new(&ode, t0, tf, y0);

    match logistic_growth_problem
        .even(1.0)
        .event(&ode)
        .solve(&mut method)
    {
        Ok(solution) => {
            if let Status::Interrupted = solution.status {
                println!("Solver stopped due to event detection");
            }

            println!("Solution points:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y);
            }

            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
        }
        Err(e) => panic!("Error: {:?}", e),
    }
}
```

### Oscillator with Amplitude Monitoring

```rust
use differential_equations::prelude::*;
use nalgebra::Vector2;

struct OscillatorEvent {
    amplitude_threshold: f64,
}

impl Event<f64, Vector2<f64>> for OscillatorEvent {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Both, Some(3)) // Detect 3 crossings
    }

    fn event(&self, _t: f64, y: &Vector2<f64>) -> f64 {
        // Event when position crosses amplitude threshold
        y[0].abs() - self.amplitude_threshold
    }
}

struct HarmonicOscillator;

impl ODE<f64, Vector2<f64>> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        dydt[0] = y[1];       // dx/dt = v
        dydt[1] = -y[0];      // dv/dt = -x (no damping)
    }
}

fn main() {
    let oscillator = HarmonicOscillator;
    let event_detector = OscillatorEvent { amplitude_threshold: 0.8 };
    
    let y0 = vector![1.0, 0.0]; // Start at maximum displacement
    let problem = ODEProblem::new(oscillator, 0.0, 20.0, y0);
    
    let solution = problem
        .dense(5)
        .event(&event_detector)
        .solve(&mut ExplicitRungeKutta::dop853())
        .unwrap();
    
    println!("Detected {} amplitude crossings", 
             solution.t.len() - 1); // -1 for initial point
}
```