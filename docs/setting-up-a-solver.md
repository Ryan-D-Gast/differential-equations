# Setting Up a Solver

This guide explains how to select, configure, and use numerical solvers for differential equations in this library.

## Overview

The library provides various numerical methods for solving different types of differential equations. Each solver implements a variant of the `NumericalMethod` trait, which provides a standard interface for initializing and stepping through the solution process.

## Selecting a Solver

When choosing a solver, consider the following factors:

1. **Equation Type** - Different solvers are optimized for different equation types (ODE, DDE, SDE).
2. **Stiffness** - Stiff equations require implicit solvers or solvers with good stability properties.
3. **Accuracy Requirements** - Higher-order methods generally provide better accuracy but may require more computational resources.
4. **Efficiency** - Some solvers are more efficient for specific problem types.

### Common Solvers

Here's a quick reference for some of the available solvers:

| Solver | Order | Type | Adaptive | Dense Output | Best For |
|--------|-------|------|----------|--------------|----------|
| RK4    | 4     | Explicit | No | No | Simple non-stiff problems |
| DOPRI5 | 5(4)  | Explicit | Yes | Yes | General-purpose non-stiff problems |
| DOP853 | 8(5)  | Explicit | Yes | Yes | High-precision non-stiff problems |
| Gauss-Legendre 6 | 6 | Implicit | Yes | Yes | Stiff problems (A-stable) |
| RKF45  | 5(4)  | Explicit | Yes | Yes | General adaptive problems |

## Basic Solver Setup

Most solvers can be instantiated and configured using a builder pattern:

```rust
use differential_equations::methods::ExplicitRungeKutta;

// Create a solver with default settings
let mut solver = ExplicitRungeKutta::dopri5();

// Or, create and configure a solver
let mut solver = ExplicitRungeKutta::dopri5()
    .rtol(1e-6)      // Relative tolerance
    .atol(1e-9)      // Absolute tolerance
    .max_steps(10000); // Maximum number of steps
```

## Common Configuration Options

Most numerical methods share these common configuration options:

### Tolerances

Tolerances control the accuracy of the adaptive step size mechanism:

```rust
let mut solver = ExplicitRungeKutta::dopri5()
    .rtol(1e-6)  // Relative tolerance
    .atol(1e-9); // Absolute tolerance
```

- **Relative tolerance (`rtol`)**: Controls the relative error. Higher values (e.g., 1e-3) allow for larger errors when state values are large.
- **Absolute tolerance (`atol`)**: Controls the absolute error. Important when state values approach zero.

### Step Size Control

```rust
let mut solver = ExplicitRungeKutta::dopri5()
    .h0(0.01)     // Initial step size
    .h_min(1e-6)  // Minimum step size
    .h_max(0.1);  // Maximum step size
```

- **Initial step size (`h0`)**: Starting step size. If set to zero, the solver will calculate an appropriate initial step size.
- **Minimum step size (`h_min`)**: Smallest allowed step size. Prevents the solver from taking extremely small steps.
- **Maximum step size (`h_max`)**: Largest allowed step size. Helps maintain accuracy over intervals where the solution changes slowly.

### Other Common Settings

```rust
let mut solver = ExplicitRungeKutta::dopri5()
    .max_steps(100000)  // Maximum number of steps
    .n_stiff(1000);     // Check for stiffness every n_stiff steps
```

- **Maximum steps (`max_steps`)**: Prevents infinite loops in case of convergence issues.

## Solver-Specific Settings

Each solver may have additional parameters specific to its algorithm. For example, DOPRI5 has parameters for step size control:

```rust
let mut solver = ExplicitRungeKutta::dopri5()
    .safe(0.9)  // Safety factor for step size control
    .beta(0.04) // Parameter for step size stabilization
    .fac1(0.2)  // Minimum factor for step size reduction
    .fac2(10.0); // Maximum factor for step size increase
```

## Using a Solver with a Problem

Once configured, a solver can be used with an `ODEProblem`:

```rust
use differential_equations::prelude::*;

// Define your ODE implementation (see Defining a Differential Equation)
struct MyODE;
impl ODE for MyODE {
    // Implementation details...
}

// Create an ODE problem
let problem = ODEProblem::new(
    MyODE,
    0.0,      // Initial time (t0)
    10.0,     // Final time (tf)
    1.0       // Initial state (y0)
);

// Configure solver
let mut solver = ExplicitRungeKutta::dopri5().rtol(1e-6).atol(1e-9);

// Solve the problem
let solution = problem.solve(&mut solver).unwrap();
```

## Dense Output

Many solvers support dense output, which allows for accurate interpolation between steps. This is particularly useful for:

1. Generating solution points at specific times
2. Event detection
3. Visualization at regular intervals

Solvers with dense output implement the `Interpolation` trait:

```rust
// Get solution at specific time point between steps
let t_interp = 2.5;
match solver.interpolate(t_interp) {
    Ok(y_interp) => println!("Interpolated value at t={}: {:?}", t_interp, y_interp),
    Err(e) => println!("Interpolation error: {:?}", e),
}
```

## Handling Stiff Problems

Stiff differential equations have components that evolve at dramatically different rates. For stiff problems:

1. Choose an appropriate solver like `Radau5` or `GuassLegendre6`
2. Implement the `jacobian` method in your ODE trait implementation
3. Consider increasing the tolerance or adjusting step size parameters

```rust
impl ODE<f64, Vector2<f64>> for StiffSystem {
    // diff implementation...

    fn jacobian(&self, t: f64, y: &Vector2<f64>, j: &mut DMatrix<f64>) {
        // Fill the jacobian matrix with partial derivatives
        j[(0, 0)] = -1000.0;               // ∂f₁/∂y₁
        j[(0, 1)] = 1.0;                   // ∂f₁/∂y₂
        j[(1, 0)] = 1.0;                   // ∂f₂/∂y₁
        j[(1, 1)] = -1.0;                  // ∂f₂/∂y₂
    }
}
```

## Example: Solving a Simple Harmonic Oscillator

Here's a complete example of setting up a solver for a harmonic oscillator:

```rust
use differential_equations::prelude::*;
use nalgebra::{Vector2, vector};

// Define a simple harmonic oscillator dx/dt = v, dv/dt = -x
struct HarmonicOscillator;

impl ODE<f64, Vector2<f64>> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
        dydt[0] = y[1];         // dx/dt = v
        dydt[1] = -y[0];        // dv/dt = -x
    }
}

fn main() {
    // Create solver with specific settings
    let mut solver = ExplicitRungeKutta::dopri5()
        .rtol(1e-6)
        .atol(1e-9)
        .max_steps(10000);

    // Initial conditions: x=1, v=0
    let y0 = vector![1.0, 0.0];
    let t0 = 0.0;
    let tf = 10.0;

    // Create and solve the problem
    let oscillator_problem = ODEProblem::new(HarmonicOscillator, t0, tf, y0);
    
    match oscillator_problem
        .even(0.1)  // Output at even intervals of 0.1
        .solve(&mut solver) 
    {
        Ok(solution) => {
            // Process solution
            for (t, y) in solution.iter().take(5) {
                println!("t = {:.2}, x = {:.6}, v = {:.6}", t, y[0], y[1]);
            }
            println!("...");
            
            // Print statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Accepted steps: {}", solution.steps.accepted);
        },
        Err(e) => println!("Error: {:?}", e),
    }
}
```

## Best Practices

1. **Start with default settings** - The default settings are chosen to work well for a wide range of problems.
2. **Adjust tolerances gradually** - If more accuracy is needed, decrease tolerances by factors of 10.
3. **Monitor function evaluations** - If performance is an issue, check the number of function evaluations.
4. **Watch for stiffness warnings** - If the solver reports stiffness, consider switching to a stiff solver.
5. **Use appropriate solvers for the problem type**:
   - Non-stiff problems: DOPRI5, DOP853, RK4
   - Stiff problems: Radau5, GuassLegendre6

## Solver Performance Considerations

Different solvers have different performance characteristics:

1. **Memory usage** - Higher order methods typically require more memory for stage values
2. **Computational cost per step** - Implicit methods are more expensive per step but can take larger steps for stiff problems
3. **Step size adaptation overhead** - Adaptive solvers require extra computation to estimate errors
4. **Dense output overhead** - Dense output requires additional storage and computation

For time-critical applications, benchmark different solvers to find the best one for your specific problem.

## Troubleshooting

Common issues and solutions:

1. **Solver taking too many steps**: The problem might be stiff or have singularities. Try a stiff solver or adjust tolerances.
2. **Step size becoming too small**: Check for discontinuities in your equation or its derivatives.
4. **Unexpected behavior near t = 0**: Make sure initial conditions are consistent with the differential equation.
