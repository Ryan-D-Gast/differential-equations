# Solout Control

Solution output control (Solout) is a powerful feature that allows you to determine how and when solution points are generated during the numerical integration process. This guide explains how to use the built-in Solout implementations and how to create your own custom implementations.

## Overview

The `Solout` trait defines how solution points are recorded during the integration process. It provides fine-grained control over:

- Which points are included in the solution
- When and how interpolation is performed
- Event detection and handling
- Special point detection (zero crossings, maxima/minima, etc.)
- Data collection for analysis

## The Solout Trait

At its core, the `Solout` trait has a single required method:

```rust
pub trait Solout<T, Y>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    fn solout<I>(
        &mut self,
        t_curr: T,      // Current time point
        t_prev: T,      // Previous time point
        y_curr: &Y,     // Current state vector
        y_prev: &Y,     // Previous state vector
        interpolator: &mut I,  // Interpolator for dense output
        solution: &mut Solution<T, Y>,  // Solution to store points in
    ) -> ControlFlag<T, Y>  // Flag to continue or terminate integration
    where
        I: Interpolation<T, Y>;
}
```

This method is called after each successful step taken by the solver. It receives:
- The current and previous time points
- The current and previous state vectors
- An interpolator that can compute the solution at any point between the previous and current times
- A mutable reference to the solution structure where points can be stored

It returns a `ControlFlag` that indicates whether the integration should continue or terminate.

## Built-in Solout Implementations

The library provides several built-in Solout implementations for common use cases:

### DefaultSolout

The simplest implementation, `DefaultSolout` stores solution points at each solver step without any interpolation. This is the default behavior if no specific Solout is provided.

```rust
use differential_equations::prelude::*;
use differential_equations::solout::DefaultSolout;

// Explicitly using DefaultSolout
let mut default_output = DefaultSolout::new();
let problem = ODEProblem::new(system, t0, tf, y0);
let solution = problem.solout(&mut default_output).solve(&mut solver).unwrap();

// Equivalent to the default behavior
let solution = problem.solve(&mut solver).unwrap();
```

### EvenSolout

`EvenSolout` generates solution points at evenly spaced time intervals, regardless of the actual steps taken by the solver:

```rust
let dt = 0.1;  // Fixed time interval
let solution = problem.even(dt).solve(&mut solver).unwrap();
```

Under the hood, this creates an `EvenSolout` instance that:
1. Determines the next time point to output (t0, t0+dt, t0+2dt, ...)
2. Uses the interpolator to calculate the state at that time point
3. Adds the interpolated point to the solution

### DenseSolout

`DenseSolout` adds multiple interpolated points between each solver step, creating a denser output:

```rust
// Generate 9 additional points between each solver step (10 total per interval)
let solution = problem.dense(10).solve(&mut solver).unwrap();
```

This is particularly useful for producing smooth plots or animations of the solution.

### TEvalSolout

`TEvalSolout` evaluates the solution at specific user-defined time points:

```rust
// Evaluate at specific time points
let evaluation_points = vec![0.0, 0.5, 1.0, 2.0, 3.14, 5.0, 7.5, 10.0];
let solution = problem.t_eval(evaluation_points).solve(&mut solver).unwrap();
```

This is useful when you need to compare the solution with data at specific time points or when you need solution values at irregular intervals.

### CrossingSolout

`CrossingSolout` detects and records when a specific component of the state vector crosses a threshold value:

```rust
use differential_equations::solout::{CrossingSolout, CrossingDirection};

// Detect when component 0 crosses zero (from any direction)
let mut crossing_detector = CrossingSolout::new(0, 0.0);

// Or only detect positive crossings (from negative to positive)
let mut positive_crossings = CrossingSolout::new(0, 0.0)
    .with_direction(CrossingDirection::Positive);

// Use it with a problem
let solution = problem.solout(&mut crossing_detector).solve(&mut solver).unwrap();

// Or use the convenience method
let solution = problem
    .crossing(0, 0.0, CrossingDirection::Both)
    .solve(&mut solver).unwrap();
```

### HyperplaneCrossingSolout

`HyperplaneCrossingSolout` detects when the trajectory crosses a hyperplane in the state space:

```rust
use differential_equations::prelude::*;
use nalgebra::{Vector3, vector};

// Define the hyperplane: y = 0 (the x-z plane)
let plane_point = vector![0.0, 0.0, 0.0];  // Any point on the plane
let plane_normal = vector![0.0, 1.0, 0.0]; // Normal vector

// Function to extract position from state vector
fn extract_position(state: &Vector6<f64>) -> Vector3<f64> {
    vector![state[0], state[1], state[2]]  // Extract position components
}

// Solve and get only the plane crossing points
let solution = problem
    .hyperplane_crossing(plane_point, plane_normal, extract_position, CrossingDirection::Both)
    .solve(&mut solver)
    .unwrap();
```

This is particularly useful for Poincaré section analysis and detecting orbital events.

## Creating Custom Solout Implementations

You can create your own custom `Solout` implementations to handle specific needs. This gives you complete control over which points are included in the solution and how events are detected.

### Example: Custom Output with Energy Calculation

Here's an example of a custom Solout implementation that adds solution points at regular intervals and also calculates and stores the energy of the system:

```rust
use differential_equations::prelude::*;

// Custom Solout implementation that tracks energy
struct EnergyTrackingSolout<T: Real> {
    dt: T,                     // Time interval
    last_output_t: Option<T>,  // Last output time
    energies: Vec<T>,          // Store energy values directly
    times: Vec<T>,             // Store time points for the energies
}

impl<T: Real> EnergyTrackingSolout<T> {
    fn new(dt: T) -> Self {
        Self {
            dt,
            last_output_t: None,
            energies: Vec::new(),
            times: Vec::new(),
        }
    }
    
    // Calculate energy for a simple harmonic oscillator
    fn calculate_energy(&self, y: &Vector2<T>) -> T {
        // Kinetic energy: 0.5 * m * v²
        let kinetic = T::from_f64(0.5).unwrap() * y[1] * y[1];
        
        // Potential energy: 0.5 * k * x²
        let potential = T::from_f64(0.5).unwrap() * y[0] * y[0];
        
        kinetic + potential
    }
    
    // Getter for the energies
    fn get_energies(&self) -> &[T] {
        &self.energies
    }
    
    // Getter for the times
    fn get_times(&self) -> &[T] {
        &self.times
    }
}

impl<T: Real> Solout<T, Vector2<T>> for EnergyTrackingSolout<T> {
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Vector2<T>,
        _y_prev: &Vector2<T>,
        interpolator: &mut I,
        solution: &mut Solution<T, Vector2<T>>,
    ) -> ControlFlag<String>
    where
        I: Interpolation<T, Vector2<T>>,
    {
        // Determine next output time
        let next_output_t = match self.last_output_t {
            Some(t) => t + self.dt,
            None => t_prev, // First call, start at t_prev
        };
        
        // Check if we need to output a point
        if next_output_t <= t_curr {
            // Interpolate at the output time
            let y_out = interpolator.interpolate(next_output_t).unwrap();
            
            // Calculate and store energy
            let energy = self.calculate_energy(&y_out);
            self.energies.push(energy);
            self.times.push(next_output_t);
            
            // Add point to solution
            solution.push(next_output_t, y_out);
            
            // Update last output time
            self.last_output_t = Some(next_output_t);
        }
        
        // Always add the current point
        if t_curr > t_prev {
            // Calculate energy at the current point
            let energy = self.calculate_energy(y_curr);
            self.energies.push(energy);
            self.times.push(t_curr);
            
            solution.push(t_curr, *y_curr);
            
            if self.last_output_t.is_none() {
                self.last_output_t = Some(t_curr);
            }
        }
        
        ControlFlag::Continue
    }
}

// Using the custom Solout
fn main() {
    // Create a harmonic oscillator
    struct HarmonicOscillator;
    
    impl ODE<f64, Vector2<f64>> for HarmonicOscillator {
        fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
            dydt[0] = y[1];       // dx/dt = v
            dydt[1] = -y[0];      // dv/dt = -x
        }
    }
    
    // Create solver and problem
    let mut solver = DOPRI5::new().rtol(1e-8).atol(1e-8);
    let y0 = vector![1.0, 0.0];  // Initial position and velocity
    let t0 = 0.0;
    let tf = 10.0;
    let problem = ODEProblem::new(HarmonicOscillator, t0, tf, y0);
    
    // Create custom Solout
    let mut energy_tracker = EnergyTrackingSolout::new(0.1);
    
    // Solve with custom Solout
    let solution = problem.solout(&mut energy_tracker).solve(&mut solver).unwrap();
    
    // Access the energies directly
    let energy_values = energy_tracker.get_energies();
    let time_points = energy_tracker.get_times();
    
    println!("Calculated {} energy values", energy_values.len());
    println!("First few energy points:");
    for i in 0..5.min(energy_values.len()) {
        println!("t = {:.2}, Energy = {:.6}", time_points[i], energy_values[i]);
    }
    
    // Energy should be conserved (approximately constant)
    if !energy_values.is_empty() {
        let max_energy = energy_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_energy = energy_values.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("Energy variation: {:.2e}", max_energy - min_energy);
    }
}
```

### Example: Event Detection with Custom Solout

Custom Solout implementations are especially powerful for event detection. Here's an example that detects when a pendulum reaches its maximum height:

```rust
struct MaxHeightSolout<T: Real> {
    last_velocity: Option<T>,
}

impl<T: Real> MaxHeightSolout<T> {
    fn new() -> Self {
        Self { last_velocity: None }
    }
}

impl<T: Real> Solout<T, Vector2<T>> for MaxHeightSolout<T> {
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Vector2<T>,
        _y_prev: &Vector2<T>,
        interpolator: &mut I,
        solution: &mut Solution<T, Vector2<T>>,
    ) -> ControlFlag<String>
    where
        I: Interpolation<T, Vector2<T>>,
    {
        let velocity = y_curr[1];
        
        // Detect zero-crossing of velocity (from positive to negative)
        // which indicates the pendulum has reached maximum height
        if let Some(last_velocity) = self.last_velocity {
            if last_velocity > T::zero() && velocity <= T::zero() {
                // Use bisection to find the exact time of maximum height
                let mut t_lower = t_prev;
                let mut t_upper = t_curr;
                let mut t_mid;
                
                // Simple bisection search for the zero-crossing
                for _ in 0..10 {  // 10 iterations of bisection
                    t_mid = (t_lower + t_upper) / T::from_f64(2.0).unwrap();
                    let y_mid = interpolator.interpolate(t_mid).unwrap();
                    let v_mid = y_mid[1];
                    
                    if v_mid > T::zero() {
                        t_lower = t_mid;
                    } else {
                        t_upper = t_mid;
                    }
                }
                
                // Use final midpoint as the maximum height point
                t_mid = (t_lower + t_upper) / T::from_f64(2.0).unwrap();
                let y_max = interpolator.interpolate(t_mid).unwrap();
                
                println!("Maximum height detected at t = {}, height = {}", t_mid, y_max[0]);
                solution.push(t_mid, y_max);
            }
        }
        
        // Update last velocity
        self.last_velocity = Some(velocity);
        
        // Always include the current point
        solution.push(t_curr, *y_curr);
        
        ControlFlag::Continue
    }
}
```

## Combining Multiple Solout Behaviors

Sometimes you might want to combine different Solout behaviors. While the library doesn't provide a direct way to compose Solout implementations, you can create a custom implementation that combines multiple behaviors:

```rust
struct CombinedSolout<T: Real> {
    even_dt: T,
    last_output_t: Option<T>,
    detect_zero_crossings: bool,
    last_value: Option<T>,
}

impl<T: Real> CombinedSolout<T> {
    fn new(dt: T, detect_crossings: bool) -> Self {
        Self {
            even_dt: dt,
            last_output_t: None,
            detect_zero_crossings: detect_crossings,
            last_value: None,
        }
    }
}

impl<T: Real> Solout<T, Vector2<T>> for CombinedSolout<T> {
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Vector2<T>,
        _y_prev: &Vector2<T>,
        interpolator: &mut I,
        solution: &mut Solution<T, Vector2<T>>,
    ) -> ControlFlag<String>
    where
        I: Interpolation<T, Vector2<T>>,
    {
        // Behavior 1: Even output points
        let next_output_t = match self.last_output_t {
            Some(t) => t + self.even_dt,
            None => t_prev,
        };
        
        if next_output_t <= t_curr {
            let y_out = interpolator.interpolate(next_output_t).unwrap();
            solution.push(next_output_t, y_out);
            self.last_output_t = Some(next_output_t);
        }
        
        // Behavior 2: Zero crossing detection
        if self.detect_zero_crossings {
            let current_value = y_curr[0];
            
            if let Some(last_value) = self.last_value {
                // Check for sign change (zero crossing)
                if last_value * current_value <= T::zero() && last_value != current_value {
                    // Find exact crossing point with bisection
                    // ... [bisection code as in the previous example] ...
                    
                    // For brevity, using linear interpolation instead
                    let t_zero = t_prev + (t_curr - t_prev) * last_value / (last_value - current_value);
                    let y_zero = interpolator.interpolate(t_zero).unwrap();
                    
                    solution.push(t_zero, y_zero);
                }
            }
            
            self.last_value = Some(current_value);
        }
        
        // Always include current point
        solution.push(t_curr, *y_curr);
        
        ControlFlag::Continue
    }
}
```

## Terminating Integration with Solout

Solout implementations can also terminate the integration by returning `ControlFlag::Terminate`:

```rust
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
    // Store the current point
    solution.push(t_curr, *y_curr);
    
    // Check if a condition is met to terminate integration
    if t_curr >= self.target_time || y_curr[0] > self.max_value {
        return ControlFlag::Terminate("Target reached".to_string());
    }
    
    ControlFlag::Continue
}
```

When a `ControlFlag::Terminate` is returned, the solver will stop the integration and the solution's status will be set to `Status::Interrupted` with the provided reason.

## Using the Interpolator Effectively

The interpolator provided to the `solout` method allows you to compute the solution at any point within the current step. Here are some tips for using it effectively:

1. **Check that the interpolation point is within bounds**: The interpolator only works within the current step (between `t_prev` and `t_curr`).

2. **Handle interpolation errors**: The `interpolate` method returns a `Result` that you should handle properly:

```rust
match interpolator.interpolate(t_interp) {
    Ok(y_interp) => {
        // Use interpolated value
        solution.push(t_interp, y_interp);
    },
    Err(e) => {
        eprintln!("Interpolation error: {:?}", e);
        // Handle error
    }
}
```

3. **Understand the accuracy**: The accuracy of the interpolation depends on the solver. For example, DOPRI5 provides a continuous extension that is accurate to 4th order.