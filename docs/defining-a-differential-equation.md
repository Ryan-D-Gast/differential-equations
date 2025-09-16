# Defining a Differential Equation

This guide explains how to define differential equations in this library using the `ODE` trait and related components.

## Overview

Differential equations in this library are defined by implementing traits that specify the derivative functions, event handling, and optionally jacobian matrices. The primary trait for ordinary differential equations is `ODE`.

## The ODE Trait

The `ODE` trait is generic over three types:
- `T`: The independent variable type (typically `f64`)
- `Y`: The state type (can be `f64`, vectors, or custom state structs)
- `D`: The data type for event callbacks

Here's the basic structure:

```rust
pub trait ODE<T = f64, Y = f64>
where
    T: Real,
    Y: State<T>,
    D: CallBackData,
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y);
    
    fn event(&self, t: T, y: &Y) -> ControlFlag<T, Y> {
        ControlFlag::Continue
    }
    
    fn jacobian(&self, t: T, y: &Y, j: &mut Matrix<T>) {
        /* Finite difference approximation */
    }
}
```

## Implementing the ODE Trait

### Basic Implementation

Let's look at an example implementing the SIR (Susceptible-Infected-Recovered) epidemiological model:

```rust
struct SIRModel {
    beta: f64,       // Transmission rate
    gamma: f64,      // Recovery rate
    population: f64, // Total population
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

## Custom State Types

You can define custom state types to represent your system's variables using the `#[derive(State)]` attribute:

```rust
#[derive(State)]
struct SIRState<T> {
    susceptible: T,
    infected: T,
    recovered: T,
}

// You can add methods to your state type
impl SIRState<f64> {
    fn population(&self) -> f64 {
        self.susceptible + self.infected + self.recovered
    }
}
```

This approach offers several advantages:
- Better code readability with named components
- Type safety for complex state vectors
- Ability to add domain-specific methods to your state

## Event Handling

The `event` method lets you define conditions that should trigger the solver to stop or take special actions:

```rust
fn event(&self, _t: f64, y: &SIRState<f64>) -> ControlFlag<PopulationMonitor> {
    if y.infected < 1.0 {
        ControlFlag::Terminate(PopulationMonitor::InfectedBelowOne)
    } else {
        ControlFlag::Continue
    }
}
```

The return type `ControlFlag<T, Y>` can be:
- `ControlFlag::Continue` - Continue integration
- `ControlFlag::Terminate` - Stop integration and return the provided reason

## Custom Event Types

You can define custom types to provide detailed information about why the solver terminated:

```rust
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

## Using the Jacobian (For Stiff Systems)

For stiff systems or when using implicit solvers, you can implement the `jacobian` method:

```rust
fn jacobian(&self, t: T, y: &Y, j: &mut DMatrix<T>) {
    // Fill the jacobian matrix j with partial derivatives
    // For a system y' = f(t,y), the jacobian is J_ij = ∂f_i/∂y_j
}
```

Note by default the jacobian is calculated using finite differences, which uses function evaluations which aren't logged.
Implementing a jacobian can result in a major performance improvement for stiff systems if the differential equation is expensive to evaluate.

## Complete Example

Here's a complete example of defining a SIR epidemiological model:

```rust
use differential_equations::{ode::methods::adams::APCV4, prelude::*};

// SIR Model definition
struct SIRModel {
    beta: f64,
    gamma: f64,
    population: f64,
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

// Custom event type
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

// Implementing the ODE trait for our model
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

## Using Vector State Types

If you prefer using vectors instead of custom state types, you can do so:

```rust
struct SimpleModel {
    parameter: f64,
}

impl ODE for SimpleModel {
    fn diff(&self, t: f64, y: &f64, dydt: &mut f64) {
        *dydt = t * (*y) + self.parameter;
    }
}

// Or with vectors
use nalgebra::Vector3;

struct VectorModel {
    parameter: f64,
}

impl ODE<f64, Vector3<f64>> for VectorModel {
    fn diff(&self, t: f64, y: &Vector3<f64>, dydt: &mut Vector3<f64>) {
        dydt[0] = y[1];
        dydt[1] = y[2];
        dydt[2] = -y[0] - y[1] + self.parameter * t;
    }
}
```