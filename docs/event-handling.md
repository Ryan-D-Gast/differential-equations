# Event Handling

Event handling is a powerful feature that allows you to detect specific conditions during the numerical integration of differential equations. This guide explains how to implement and use event functions effectively in this library.

## Overview

Events in differential equation solvers serve multiple purposes:

1. **Termination conditions**: Stop the integration when a specific condition is met
2. **State monitoring**: Track when the system reaches critical thresholds  
3. **Discontinuity handling**: Detect and handle discontinuities in the solution
4. **Special point detection**: Record when the system passes through points of interest

This library supports robust event handling through the `event` method in the differential equation traits (e.g., `ODE`, `DDE`) and flexible callback data types.

## Basic Event Handling

The most straightforward way to implement event handling is through the `event` method in your differential equation implementation:

```rust
impl ODE for MySystem {
    fn diff(&self, t: f64, y: &f64, dydt: &mut f64) {
        // Differential equation implementation
        *dydt = /* ... */;
    }

    fn event(&self, t: f64, y: &f64) -> ControlFlag<String> {
        if *y > 10.0 {
            // Integration will stop when y exceeds 10.0
            ControlFlag::Terminate("y exceeded threshold".to_string())
        } else {
            // Continue integration
            ControlFlag::Continue
        }
    }
}
```

The `event` method is called after each successful step of the numerical method. It receives the current time `t` and state `y`, and returns a `ControlFlag` indicating whether to continue or terminate the integration.

## The `ControlFlag` Enum

The `ControlFlag` enum has two variants:

1. `ControlFlag::Continue` - Continue the integration
2. `ControlFlag::Terminate(data)` - Stop the integration and return the provided data

The data type of the termination reason is generic, allowing you to use any type that implements the `CallBackData` trait.

## Using String Callbacks (Default)

The simplest approach is to use `String` as your callback data type:

```rust
fn event(&self, _t: f64, y: &f64) -> ControlFlag<String> {
    if *y > 10.0 {
        ControlFlag::Terminate("y exceeded threshold".to_string())
    } else {
        ControlFlag::Continue
    }
}
```

This is the default if you don't specify a callback type when implementing the `ODE` trait:

```rust
// Using the default String callback type
impl ODE for LogisticGrowth {
    // Implementation...
}
```

### Example: Logistic Growth Model

Here's a complete example of using string callbacks with a logistic growth model:

```rust
struct LogisticGrowth {
    k: f64,  // Growth rate
    m: f64,  // Carrying capacity
}

impl ODE for LogisticGrowth {
    fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = self.k * y * (1.0 - y / self.m);
    }

    fn event(&self, _t: f64, y: &f64) -> ControlFlag<String> {
        if *y > 0.9 * self.m {
            // Stop when population reaches 90% of carrying capacity
            ControlFlag::Terminate("Reached 90% of carrying capacity".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}
```

## Custom Callback Types

For more structured and type-safe event handling, you can define custom types to represent different termination conditions:

```rust
// Custom enum for different termination conditions
#[derive(Debug, Clone)]
enum PopulationMonitor {
    InfectedBelowOne,
    PopulationDiedOut,
}

// Implement Display for better error messages
impl std::fmt::Display for PopulationMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PopulationMonitor::InfectedBelowOne => write!(f, "Infected population is below 1"),
            PopulationMonitor::PopulationDiedOut => write!(f, "Population died out"),
        }
    }
}
```

When using custom callback types, you need to explicitly specify them when implementing the ODE trait:

```rust
impl ODE<f64, SIRState<f64>, PopulationMonitor> for SIRModel {
    // Implementation...
    
    fn event(&self, _t: f64, y: &SIRState<f64>) -> ControlFlag<PopulationMonitor> {
        if y.infected < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::InfectedBelowOne)
        } else if y.population() < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::PopulationDiedOut)
        } else {
            ControlFlag::Continue
        }
    }
}
```

### Example: SIR Epidemiological Model

Here's a complete example using custom callback types with the SIR (Susceptible-Infected-Recovered) model:

```rust
/// SIR Model
struct SIRModel {
    beta: f64,       // Transmission rate
    gamma: f64,      // Recovery rate
    population: f64, // Total population
}

#[derive(Debug, Clone)]
enum PopulationMonitor {
    InfectedBelowOne,
    PopulationDiedOut,
}

impl std::fmt::Display for PopulationMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PopulationMonitor::InfectedBelowOne => write!(f, "Infected population is below 1"),
            PopulationMonitor::PopulationDiedOut => write!(f, "Population died out"),
        }
    }
}

// Custom state type with the State trait derived
#[derive(State)]
struct SIRState<T> {
    susceptible: T,
    infected: T,
    recovered: T,
}

impl SIRState<f64> {
    fn population(&self) -> f64 {
        self.susceptible + self.infected + self.recovered
    }
}

impl ODE<f64, SIRState<f64>, PopulationMonitor> for SIRModel {
    fn diff(&self, _t: f64, y: &SIRState<f64>, dydt: &mut SIRState<f64>) {
        let s = y.susceptible;
        let i = y.infected;
        
        dydt.susceptible = -self.beta * s * i / self.population;
        dydt.infected = self.beta * s * i / self.population - self.gamma * i;
        dydt.recovered = self.gamma * i;
    }

    fn event(&self, _t: f64, y: &SIRState<f64>) -> ControlFlag<PopulationMonitor> {
        if y.infected < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::InfectedBelowOne)
        } else if y.population() < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::PopulationDiedOut)
        } else {
            ControlFlag::Continue
        }
    }
}
```

## Accessing Event Results

When a solver terminates due to an event, the reason is stored in the `Status::Interrupted` variant of the solution's status:

```rust
match problem.solve(&mut method) {
    Ok(solution) => {
        if let Status::Interrupted(ref reason) = solution.status {
            println!("Solver stopped: {}", reason);
            // For custom types, you can pattern match
            if let Some(PopulationMonitor::InfectedBelowOne) = reason.downcast_ref() {
                println!("The infection has been contained!");
            }
        }
        // Process solution...
    },
    Err(e) => panic!("Error: {:?}", e),
};
```

## Root-Finding for Event Detection

When an event is triggered, the solver performs root-finding to accurately locate the time and state at which the event occurred. This ensures that events are detected precisely, not just at the solver's integration steps.

The root-finding algorithm:

1. Detects when the event function changes from `Continue` to `Terminate` between steps
2. Uses interpolation to find the exact time where the event occurs
3. Updates the final time and state to the event point
4. Returns the appropriate callback data

## Advanced Event Handling

### Multiple Conditions

You can check multiple conditions within a single event function:

```rust
fn event(&self, t: f64, y: &MyState<f64>) -> ControlFlag<MyEvents> {
    if y.temperature > self.critical_temperature {
        ControlFlag::Terminate(MyEvents::TemperatureExceeded)
    } else if y.pressure > self.max_pressure {
        ControlFlag::Terminate(MyEvents::PressureExceeded)
    } else if t > self.time_limit {
        ControlFlag::Terminate(MyEvents::TimeExceeded)
    } else {
        ControlFlag::Continue
    }
}
```

### Combining with Solout

For more complex event handling scenarios, you can implement custom `Solout` implementations that return appropriate control flags:

```rust
use differential_equations::{ControlFlag, Solution, prelude::*};

// Define a custom Solout implementation
struct CustomEventSolout<T: Real> {
    threshold: T,
    event_happened: bool,
}

impl<T: Real> CustomEventSolout<T> {
    fn new(threshold: T) -> Self {
        Self {
            threshold,
            event_happened: false,
        }
    }
}

// Implement the Solout trait for our custom solout
impl<T: Real, Y: State<T>> Solout<T, Y> for CustomEventSolout<T> {
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Y,
        y_prev: &Y,
        interpolator: &mut I,
        solution: &mut Solution<T, Y>,
    ) -> ControlFlag<String>
    where
        I: Interpolation<T, Y>,
    {
        // Add the current point to the solution
        solution.push(t_curr, *y_curr);
        
        // Check for our event condition using both current and interpolated values
        if !self.event_happened {
            // Get a value from the state (e.g., first component for vector states)
            let value = if y_curr.len() > 0 { y_curr.get(0) } else { *y_curr };
            
            // Check if our threshold was crossed
            if value > self.threshold {
                self.event_happened = true;
                
                // Find the exact crossing point through interpolation
                let mut t_event = t_prev;
                let mut t_step = (t_curr - t_prev) / T::from_f64(10.0).unwrap();
                let mut event_found = false;
                
                // Simple bisection search for the event point
                while t_event < t_curr && !event_found {
                    let y_interp = interpolator.interpolate(t_event).unwrap();
                    let interp_value = if y_interp.len() > 0 { y_interp.get(0) } else { y_interp };
                    
                    if interp_value > self.threshold {
                        event_found = true;
                        // Add the exact event point to the solution
                        solution.push(t_event, y_interp);
                        return ControlFlag::Terminate("Threshold exceeded".to_string());
                    }
                    t_event += t_step;
                }
                
                // If exact point not found, still terminate
                return ControlFlag::Terminate("Threshold exceeded".to_string());
            }
        }
        
        ControlFlag::Continue
    }
}

// Using the custom solout
fn main() {
    let problem = ODEProblem::new(/* ... */);
    let mut solver = DOPRI5::new();
    let mut custom_solout = CustomEventSolout::new(5.0); // Threshold of 5.0
    
    match problem
        .solout(&mut custom_solout)
        .solve(&mut solver)
    {
        Ok(solution) => {
            if let Status::Interrupted = solution.status {
                println!("Integration stopped: {}", reason);
            }
            // Process solution...
        },
        Err(e) => println!("Error: {:?}", e),
    }
}
```

This approach gives you complete control over:
1. When and how to interpolate between steps
2. The criteria for event detection
3. How the event points are added to the solution
4. What information is returned in the event message

### Using Built-in Output Options with Event Detection

The library's built-in output handlers like `even` or `dense` already implement the `Solout` trait and can be combined with the event function:

```rust
// Combining even output with event detection
let result = problem
    .even(0.1) // Output at evenly spaced points with dt = 0.1
    .solve(&mut solver);

// Check if solver was terminated by an event
if let Ok(solution) = result {
    if let Status::Interrupted = &solution.status {
        println!("Solver stopped due to event: {}", reason);
    }
}
```

## Example: Oscillator with Amplitude Threshold

Here's a complete example of an oscillator that terminates when the amplitude exceeds a threshold:

```rust
use differential_equations::prelude::*;
use nalgebra::{Vector2, vector};

struct OscillatorWithDamping {
    damping: f64,
    drive_strength: f64,
    drive_freq: f64,
    threshold: f64,
}

#[derive(Debug, Clone)]
enum OscillatorEvent {
    AmplitudeExceeded,
    EnergyDissipated,
}

impl std::fmt::Display for OscillatorEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OscillatorEvent::AmplitudeExceeded => write!(f, "Oscillator amplitude exceeded threshold"),
            OscillatorEvent::EnergyDissipated => write!(f, "Energy fully dissipated"),
        }
    }
}

impl ODE<f64, Vector2<f64>, OscillatorEvent> for OscillatorWithDamping {
    fn diff(&self, t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        let position = y[0];
        let velocity = y[1];
        
        // dx/dt = v
        dydt[0] = velocity;
        
        // dv/dt = -x - c*v + F*sin(w*t)  [damped, driven oscillator]
        dydt[1] = -position - self.damping * velocity + 
                  self.drive_strength * (self.drive_freq * t).sin();
    }
    
    fn event(&self, _t: f64, y: &Vector2<f64>) -> ControlFlag<OscillatorEvent> {
        let position = y[0];
        let velocity = y[1];
        
        // Calculate amplitude (position)
        if position.abs() > self.threshold {
            ControlFlag::Terminate(OscillatorEvent::AmplitudeExceeded)
        }
        // Check if energy (position^2 + velocity^2) is nearly dissipated
        else if position.powi(2) + velocity.powi(2) < 1e-6 {
            ControlFlag::Terminate(OscillatorEvent::EnergyDissipated)
        } 
        else {
            ControlFlag::Continue
        }
    }
}

fn main() {
    let oscillator = OscillatorWithDamping {
        damping: 0.1,
        drive_strength: 0.5,
        drive_freq: 1.0,
        threshold: 3.0,
    };
    
    // Initial conditions: position = 0, velocity = 1
    let y0 = vector![0.0, 1.0];
    
    // Solve from t=0 to t=50
    let mut solver = DOPRI5::new().rtol(1e-6).atol(1e-9);
    let problem = ODEProblem::new(oscillator, 0.0, 50.0, y0);
    
    match problem.dense().solve(&mut solver) {
        Ok(solution) => {
            // Check if solver terminated due to an event
            if let Status::Interrupted = &solution.status {
                println!("Solver terminated: {}", reason);
            }
            
            // Process solution...
            println!("Final time: {}", solution.last().unwrap().0);
            println!("Solution points: {}", solution.len());
        },
        Err(e) => println!("Error: {:?}", e),
    }
}
```