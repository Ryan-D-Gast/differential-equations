# Solution Handling

After successfully solving a differential equation, the result is returned as a `Solution` struct. This guide explains the structure of the solution object and how to work with it effectively.

## Overview

The `Solution` struct contains the numerical solution data as well as metadata about the solving process. It provides methods for accessing, iterating over, and exporting the solution.

```rust
pub struct Solution<T, V, D>
where
    T: Real,
    V: State<T>,
    D: CallBackData,
{
    pub t: Vec<T>,             // Time points
    pub y: Vec<V>,             // State vectors at each time point
    pub status: Status<T, V, D>, // Solver status (success, error, or interrupted)
    pub evals: usize,          // Number of function evaluations
    pub jac_evals: usize,      // Number of Jacobian evaluations
    pub steps: usize,          // Total number of steps taken
    pub rejected_steps: usize, // Number of rejected steps
    pub accepted_steps: usize, // Number of accepted steps
    pub timer: Timer<T>,       // Timer for measuring solving time
}
```

## Accessing Solution Data

### Direct Field Access

You can directly access the solution data through its public fields:

```rust
// Access time and state vectors
let times = &solution.t;
let states = &solution.y;

// Get performance statistics
println!("Function evaluations: {}", solution.evals);
println!("Steps taken: {}", solution.steps);
println!("Rejected steps: {}", solution.rejected_steps);
println!("Accepted steps: {}", solution.accepted_steps);
println!("Solve time: {:?} seconds", solution.timer.elapsed());
```

### Using Methods

The `Solution` struct provides several convenient methods for accessing its data:

#### Getting the Last Point

The `last()` method returns a reference to the last time point and state vector:

```rust
match solution.last() {
    Ok((t, y)) => {
        println!("Final time: {}", t);
        println!("Final state: {:?}", y);
    },
    Err(e) => println!("Error retrieving last point: {}", e),
}
```

#### Iterating Over Solutions

The `iter()` method returns an iterator that yields pairs of time points and state vectors:

```rust
// Print all solution points
println!("Time, State");
for (t, y) in solution.iter() {
    println!("{}, {:?}", t, y);
}
```

#### Converting to a Tuple

If you only need the raw data and not the metadata, you can convert the solution to a tuple of vectors:

```rust
let (t_vec, y_vec) = solution.into_tuple();
```

Note that this consumes the solution object, so you won't be able to access it afterward.

## Checking Solution Status

The `status` field indicates whether the solution was successful, encountered an error, or was interrupted by an event:

```rust
match solution.status {
    Status::Completed => {
        println!("Solver completed successfully");
    },
    Status::Interrupted(ref reason) => {
        println!("Solver was interrupted: {}", reason);
        
        // For custom event types, you can downcast
        if let Some(event) = reason.downcast_ref::<MyEventEnum>() {
            match event {
                MyEventEnum::ThresholdExceeded => {
                    // Handle specific event
                },
                // Other event variants...
            }
        }
    },
    Status::Error(ref error) => {
        println!("Solver encountered an error: {:?}", error);
    },
    _ => println!("Solver status: {:?}", solution.status),
}
```

## Exporting Solutions

### CSV Export

The `to_csv()` method allows you to export the solution to a CSV file:

```rust
match solution.to_csv("results/my_solution.csv") {
    Ok(_) => println!("Solution saved to CSV"),
    Err(e) => println!("Failed to save solution: {}", e),
}
```

The CSV will contain columns for the independent variable (t) and each component of the state vector (y0, y1, ...). If the directory doesn't exist, it will be created automatically.

### Polars Integration

If the `polars` feature is enabled, you can convert the solution to a Polars DataFrame:

```rust
#[cfg(feature = "polars")]
match solution.to_polars() {
    Ok(df) => {
        println!("Converted to DataFrame:");
        println!("{:?}", df);
        
        // Perform operations using polars
        let filtered = df.filter(&df["t"].gt(5.0)).unwrap();
        println!("Points after t=5: {}", filtered.height());
    },
    Err(e) => println!("Error creating DataFrame: {}", e),
}
```

## Example: Processing a Solution

Here's a complete example of solving an ODE and processing the solution:

```rust
use differential_equations::prelude::*;
use nalgebra::{Vector2, vector};

// Define ODE system (damped harmonic oscillator)
struct DampedOscillator {
    damping: f64,
}

impl ODE<f64, Vector2<f64>> for DampedOscillator {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        dydt[0] = y[1];
        dydt[1] = -y[0] - self.damping * y[1];
    }
}

fn main() {
    // Create the system and solver
    let system = DampedOscillator { damping: 0.1 };
    let mut solver = DOPRI5::new().rtol(1e-6).atol(1e-8);
    
    // Initial conditions: position = 1.0, velocity = 0.0
    let y0 = vector![1.0, 0.0];
    let t0 = 0.0;
    let tf = 20.0;
    
    // Solve
    let problem = ODEProblem::new(system, t0, tf, y0);
    match problem.even(0.1).solve(&mut solver) {
        Ok(solution) => {
            // Print summary statistics
            println!("Solution contains {} points", solution.t.len());
            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {} (accepted: {}, rejected: {})",
                solution.steps, solution.accepted_steps, solution.rejected_steps);
            println!("Solve time: {:?} seconds", solution.timer.elapsed());
            
            // Calculate maximum position
            let max_pos = solution.iter()
                .map(|(_, y)| y[0].abs())
                .fold(f64::NEG_INFINITY, f64::max);
            println!("Maximum position: {:.6}", max_pos);
            
            // Calculate time to decay to 10% of initial amplitude
            for (t, y) in solution.iter() {
                if y[0].abs() < 0.1 {
                    println!("Decayed to 10% at t = {:.6}", t);
                    break;
                }
            }
            
            // Calculate energy at each point
            let energies: Vec<f64> = solution.iter()
                .map(|(_, y)| {
                    let kinetic = 0.5 * y[1] * y[1];
                    let potential = 0.5 * y[0] * y[0];
                    kinetic + potential
                })
                .collect();
                
            println!("Initial energy: {:.6}", energies[0]);
            println!("Final energy: {:.6}", energies.last().unwrap());
            println!("Energy loss: {:.6}%", 
                (1.0 - energies.last().unwrap() / energies[0]) * 100.0);
            
            // Save to CSV
            if let Err(e) = solution.to_csv("results/damped_oscillator.csv") {
                println!("Failed to save CSV: {}", e);
            }
        },
        Err(e) => println!("Error solving ODE: {:?}", e),
    }
}
```

## Advanced: Working with Custom State Types

When using custom state types with the `#[derive(State)]` attribute, you can still access the solution data using your type's methods:

```rust
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
    
    fn infection_rate(&self) -> f64 {
        self.infected / self.population()
    }
}

// After solving...
let max_infection_rate = solution.iter()
    .map(|(_, state)| state.infection_rate())
    .fold(f64::NEG_INFINITY, f64::max);

println!("Peak infection rate: {:.2}%", max_infection_rate * 100.0);
```

## Using the Timer

The `Solution` struct includes a timer that measures the solving time:

```rust
println!("Solution took {} seconds", solution.timer.elapsed());
```

The timer is started automatically when the solver begins integration and completed when the solver finishes.