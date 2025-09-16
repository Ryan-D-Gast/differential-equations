# Delay Differential Equations (DDE)

The `dde` module provides tools for solving delay differential equations (DDEs), which are differential equations where the derivative of the unknown function at a certain time depends on the solution at previous times. This module focuses on initial value problems for DDEs (`DDEProblem`).

## Table of Contents

- [Defining a DDE](#defining-a-dde)
- [The History Function](#the-history-function)
- [Solving an Initial Value Problem (DDEProblem)](#solving-an-initial-value-problem)
- [Examples](#examples)
- [Notation](#notation)

## Defining a DDE

The `DDE` trait defines the delay differential equation `dydt = f(t, y(t), y(t - lag_1), ..., y(t - lag_L))` and the lag functions.

### DDE Trait
*   `diff(&self, t: T, y: &Y, yd: &[Y; L], dydt: &mut Y)`: Defines the differential equation. `y` is the current state `y(t)`, and `yd` is an array slice containing the delayed states `[y(t - lag_1), ..., y(t - lag_L)]`. `L` is a const generic parameter indicating the number of constant/state-dependent lags.
*   `lags(&self, t: T, y: &Y, lags: &mut [T; L])`: Defines the actual time lag values `[lag_1, ..., lag_L]` at time `t` and state `y`. These lags determine the past times `t - lag_i` at which the solution is evaluated.
*   `event(&self, t: T, y: &Y) -> ControlFlag<T, Y>`: Optional event function, similar to the ODE trait, to interrupt the solver. `D` is the type for event data, defaulting to `String`. By default, it returns `ControlFlag::Continue`.

### Solout Trait
The `Solout` trait works similarly to how it does for ODEs, allowing customization of which points are saved in the solution.

### Implementation Example
```rust
use differential_equations::prelude::*;
use nalgebra::Vector3; // Using nalgebra for state vector

// Example: Breast Cancer Model with Delay
// du₁/dt = (v₀ / (1 + β₀ * u₃(t-τ)²)) * (p₀ - q₀) * u₁ - d₀ * u₁
// du₂/dt = (v₀ / (1 + β₀ * u₃(t-τ)²)) * (1 - p₀ + q₀) * u₁ + (v₁ / (1 + β₁ * u₃(t-τ)²)) * (p₁ - q₁) * u₂ - d₁ * u₂
// du₃/dt = (v₁ / (1 + β₁ * u₃(t-τ)²)) * (1 - p₁ + q₁) * u₂ - d₂ * u₃
struct BreastCancerModel {
    p0: f64, q0: f64, v0: f64, d0: f64,
    p1: f64, q1: f64, v1: f64, d1: f64, d2: f64,
    beta0: f64, beta1: f64,
    tau: f64, // The time delay
}

// L=1 because there is one delay term, Y is Vector3<f64>
impl DDE<1, f64, Vector3<f64>> for BreastCancerModel {
    fn diff(&self, _t: f64, u: &Vector3<f64>, ud: &[Vector3<f64>; 1], dudt: &mut Vector3<f64>) {
        // ud[0] corresponds to u(t - lags[0])
        // ud[0][2] is u₃(t-τ)
        let hist3 = ud[0][2];

        let term0_common = self.v0 / (1.0 + self.beta0 * hist3.powi(2));
        let term1_common = self.v1 / (1.0 + self.beta1 * hist3.powi(2));

        dudt[0] = term0_common * (self.p0 - self.q0) * u[0] - self.d0 * u[0];
        dudt[1] = term0_common * (1.0 - self.p0 + self.q0) * u[0]
            + term1_common * (self.p1 - self.q1) * u[1]
            - self.d1 * u[1];
        dudt[2] = term1_common * (1.0 - self.p1 + self.q1) * u[1] - self.d2 * u[2];
    }

    fn lags(&self, _t: f64, _y: &Vector3<f64>, lags: &mut [f64; 1]) {
        // Define the constant lag
        lags[0] = self.tau;
    }

    // Optional event function
    // fn event(&self, t: f64, y: &Vector3<f64>) -> ControlFlag<T, Y> { // D defaults to String
    //     if y[0] < 0.0 { // Example: check if first component is negative
    //         ControlFlag::Terminate("u1 became negative".to_string())
    //     } else {
    //         ControlFlag::Continue
    //     }
    // }
}
```
Generics `<const L: usize, T, Y>` are used: `L` is the number of discrete lags, `T` is the float type (e.g., `f64`), `Y` is the state vector type, and `D` is the type for event data (defaulting to `String`).

## The History Function

A crucial component for solving DDEs is the **history function**. This function, `phi(t)`, provides the solution `y(t)` for all times `t <= t0`, where `t0` is the initial time of the simulation. The solver uses this function to look up values of `y` at past times when evaluating delayed terms, especially at the beginning of the integration interval.

The history function is provided when creating a `DDEProblem` and should have the signature `Fn(T) -> V`.

```rust
// Example history function: y(t) = initial_state for t <= 0
// For the BreastCancerModel with state Vector3<f64>
let initial_state = Vector3::new(1.0, 1.0, 1.0);
let history_fn = |_t: f64| -> Vector3<f64> {
    initial_state // Return the constant initial state for t <= t0
};
```
The history function takes the time `t` (where `t <= t0`) and returns the historical value of `y(t)` as type `Y`.

## Solving an Initial Value Problem (DDEProblem)

The `DDEProblem` struct is used to set up and solve the DDE. It requires the DDE system, the time interval `[t0, tf]`, the initial state `y0` (which is `y(t0)`), and the history function `phi`.

```rust
fn main() {
    let mut method = ExplicitRungeKutta::dopri5().rtol(1e-6).atol(1e-8); // Using the DDE45 (DOPRI5) solver

    let t0 = 0.0;
    let tf = 10.0;
    let y0 = Vector3::new(1.0, 1.0, 1.0); // Initial state u(t0)
    
    let system = BreastCancerModel {
        p0: 0.2, q0: 0.3, v0: 1.0, d0: 5.0,
        p1: 0.2, q1: 0.3, v1: 1.0, d1: 1.0, d2: 1.0,
        beta0: 1.0, beta1: 1.0,
        tau: 1.0, // Delay value
    };

    // History function: u(t) = y0 for t <= t0
    let history_fn = |_t: f64| -> Vector3<f64> {
        y0 // Return initial state for all t <= t0
    };

    let problem = DDEProblem::new(system, t0, tf, y0, history_fn);

    match problem
        .even(0.5) // Example: Save solution every 0.5 time units
        .solve(&mut method)
    {
        Ok(solution) => {
            if let Status::Interrupted(ref reason) = solution.status {
                println!("Solver stopped: {}", reason);
            }
            println!("Solution (Breast Cancer Model):");
            for (t, u_vec) in solution.iter() {
                println!("(t: {:.4}, u1: {:.4}, u2: {:.4}, u3: {:.4})", t, u_vec[0], u_vec[1], u_vec[2]);
            }
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
        }
        Err(e) => panic!("Error: {:?}", e),
    }
}
```
The `solve` method returns a `Result<Solution, Status>`, where `Solution` contains the time points, corresponding solution values, and solver statistics.

### Result

```
Solving Breast Cancer Model (tau=1) from t=0 to t=10...
Solver finished with status: Complete        
Function evaluations: 198
Solver steps: 28
Accepted steps: 27
Rejected steps: 1
Number of output points: 21
t: 0.0000, u1: 1.0000, u2: 1.0000, u3: 1.0000
t: 0.5000, u1: 0.0801, u2: 0.6619, u3: 0.7841
t: 1.0000, u1: 0.0064, u2: 0.3972, u3: 0.5856
t: 1.5000, u1: 0.0005, u2: 0.2348, u3: 0.4282
t: 2.0000, u1: 0.0000, u2: 0.1377, u3: 0.3123
t: 2.5000, u1: 0.0000, u2: 0.0802, u3: 0.2253
t: 3.0000, u1: 0.0000, u2: 0.0466, u3: 0.1597
t: 3.5000, u1: -0.0000, u2: 0.0270, u3: 0.1111
t: 4.0000, u1: -0.0000, u2: 0.0156, u3: 0.0758
t: 4.5000, u1: 0.0000, u2: 0.0090, u3: 0.0510
t: 5.0000, u1: -0.0000, u2: 0.0052, u3: 0.0338
t: 5.5000, u1: 0.0000, u2: 0.0030, u3: 0.0222
t: 6.0000, u1: 0.0000, u2: 0.0017, u3: 0.0144
t: 6.5000, u1: 0.0000, u2: 0.0010, u3: 0.0093
t: 7.0000, u1: -0.0000, u2: 0.0006, u3: 0.0060
t: 7.5000, u1: -0.0000, u2: 0.0003, u3: 0.0038
t: 8.0000, u1: 0.0000, u2: 0.0002, u3: 0.0024
t: 8.5000, u1: 0.0000, u2: 0.0001, u3: 0.0015
t: 9.0000, u1: -0.0000, u2: 0.0001, u3: 0.0010
t: 9.5000, u1: -0.0000, u2: 0.0000, u3: 0.0006
t: 10.0000, u1: 0.0000, u2: 0.0000, u3: 0.0004
```

## Examples

For more detailed examples of DDEs, refer to the `examples/dde` directory in the repository. These examples showcase different DDE systems, solver configurations, and usage patterns.
*   **Mackey-Glass Equation**: A common benchmark DDE exhibiting chaotic behavior.
*   **Breast Cancer Model**: A model of tumor growth with delays in the response of the immune system.

## Notation

This library uses the following notation for DDEs:
-   `t`: The independent variable, typically time.
-   `y`: The dependent state vector `y(t)`.
-   `dydt`: The derivative of `y` with respect to `t`, i.e., `y'(t)`.
-   `lags`: An array `[lag_1, ..., lag_L]` containing the time delay values.
-   `yd`: An array `[y(t - lag_1), ..., y(t - lag_L)]` containing the state vectors at delayed times.
-   `k`: Coefficients or intermediate stages within solver algorithms, often representing approximations to derivatives.

Consistency with this notation is encouraged for any future additions to the DDE module.