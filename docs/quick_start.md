# Quick Start: Defining State for ODEs

This guide provides a brief overview of how to define the state and differential equations for use with the Ordinary Differential Equation (ODE) solvers.

## The `ODE` Trait

The core of defining your problem lies in implementing the `ODE` trait. This trait defines the differential equation `dydt = f(t, y)` that the solver will use.

Key components of the `ODE` trait:

*   `diff(t, y, dydt)`: This method defines the system of differential equations.
    *   `t`: The current value of the independent variable (e.g., time).
    *   `y`: A reference to the current state vector.
    *   `dydt`: A mutable reference to the vector where the derivatives should be stored.
*   `event(t, y)`: An optional method to define conditions for interrupting the solver.
    *   It receives the current time `t` and state `y`.
    *   It should return a `ControlFlag`. By default, it returns `ControlFlag::Continue`.
    *   To terminate, return `ControlFlag::Terminate(reason)`, where `reason` can be any type implementing `Clone + Debug` (defaults to `String`).

## Example Implementation

Here's an example of implementing the `ODE` trait for a logistic growth model:

```rust
// Includes required elements and common methods.
// Less common methods are in the modules such as`ode::methods::...`
use differential_equations::prelude::*; 
use nalgebra::{SVector, vector};

// Define a struct to hold parameters for the ODE
struct LogisticGrowth {
    k: f64, // Growth rate
    m: f64, // Carrying capacity
}

// Implement the ODE trait for the LogisticGrowth struct
impl ODE<f64, SVector<f64, 1>> for LogisticGrowth {
    // Define the differential equation: dy/dt = k * y * (1 - y / m)
    fn diff(&self, t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0] * (1.0 - y[0] / self.m);
    }

    // Optional: Define an event function to stop the solver
    fn event(&self, t: f64, y: &SVector<f64, 1>) -> ControlFlag {
        if y[0] > 0.9 * self.m { // If population exceeds 90% of carrying capacity
            ControlFlag::Terminate("Reached 90% of carrying capacity".to_string())
        } else {
            ControlFlag::Continue // Otherwise, continue solving
        }
    }
}
```

## Solving an Initial Value Problem with `ODEProblem`

Once you have implemented the `ODE` trait for your system, you can use the `ODEProblem` struct to set up and solve the initial value problem. The `ODEProblem` acts as a solver controller, orchestrating the solution process.

### Setting up the Problem

You create an `ODEProblem` by providing:
*   An instance of your `ODE` implementation (e.g., `LogisticGrowth`).
*   The initial time `t0`.
*   The final time `tf`.
*   The initial state vector `y0`.

### Choosing a Numerical Method

You'll also need to select a numerical method (solver). The library offers various fixed-step and adaptive-step solvers. For example, `DOP853` is a good general-purpose adaptive solver.

### Solving and Handling Output

The `solve` method on `ODEProblem` executes the numerical integration. You can configure how the solution is outputted, for instance, by requesting points at even intervals using the `even()` method.

The `solve` method returns a `Result<Solution, Status>`. The `Solution` struct contains the time points, state vectors, and solver statistics. The `Status` indicates how the solver finished (e.g., `Ok`, `Interrupted`).

### Example: Solving Logistic Growth

Here's how to solve the `LogisticGrowth` model defined earlier:

```rust
use differential_equations::prelude::*; 
use nalgebra::{SVector, vector};

fn main() {
    // Choose a numerical method and set tolerances
    let mut method = ExplicitRungeKutta::dop853()
        .rtol(1e-7) // Set relative tolerance
        .atol(1e-7);// Set absolute tolerance

    // Define initial conditions and time span
    let y0 = vector![1.0]; // Initial population
    let t0 = 0.0;          // Start time
    let tf = 10.0;         // End time

    // Create an instance of the ODE
    let ode_system = LogisticGrowth { k: 1.0, m: 10.0 };

    // Create the ODEProblem
    let problem = ODEProblem::new(ode_system, t0, tf, y0);

    // Solve the problem, requesting output every 1.0 time unit
    match problem
        .even(1.0) // Set Solout to output every 1.0 time unit
        .solve(&mut method) // Solve the ODE
    {
        Ok(solution) => {
            // Check if the solver was interrupted by the event function
            if let Status::Interrupted(ref reason) = solution.status {
                println!("Solver stopped: {}", reason);
            }

            // Print the solution
            println!("Solution (t, y):");
            for (t, y_val) in solution.iter() {
                println!("({:.4}, {:.4})", t, y_val[0]);
            }

            // Print solver statistics
            println!("\nSolver Statistics:");
            println!("  Function evaluations: {}", solution.evals.function);
            println!("  Steps taken: {}", solution.steps.total());
            println!("  Accepted steps: {}", solution.steps.accepted);
            println!("  Rejected steps: {}", solution.steps.rejected);
        }
        Err(e) => eprintln!("An error occurred: {:?}", e),
    }
}
```

### Output

```sh
Solver stopped: Reached 90% of carrying capacity
Solution (t, y):
(0.0000, 1.0000)
(1.0000, 2.3197)
(2.0000, 4.5085)
(3.0000, 6.9057)
(4.0000, 8.5849)
(4.3944, 9.0000)

Solver Statistics:
  Function evaluations: 359
  Steps taken: 25
  Accepted steps: 21
  Rejected steps: 4
```

## Understanding Generics: `T` and `V`

The `ODE` trait and related components like `ODEProblem` are defined with generics `<T, V,...>`:

*   `T`: Represents the floating-point type used for calculations (e.g., `f64` or `f32`). This type applies to time and the components of the state vector.
*   `V`: Represents the type of the state vector. This can be:
    *   A simple float (`f64` or `f32`) for a single ODE.
    *   An `nalgebra::SVector<T, N>` for a system of N ODEs, where `N` is the number of equations.
    *   A custom struct (deriving the `State` trait) with fields of type `T`.

By default, if generics are omitted when implementing the `ODE` trait, they are assumed to be `f64` for both `T` and `V`. This implies a single ODE where both the state and time are `f64`.

In the `LogisticGrowth` example, `ODE<f64, SVector<f64, 1>>` explicitly defines `T` as `f64` and `V` as an `SVector` containing one `f64` element. While `f64` could have been used directly for `V` in this single-equation case, using `SVector<f64, 1>` provides clarity and is consistent with how systems of multiple ODEs (e.g., `SVector<f64, N>`) would be defined.

This generic setup provides flexibility, allowing you to define simple or complex systems of differential equations tailored to your specific numerical precision and state representation needs.