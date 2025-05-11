<p align="center">
  <img src="../assets/logo.svg" width="1000" alt="differential-equations">
</p>

-----

<p align="center">
<b>A high-performance library for numerically solving differential equations</b><br>
<i>for the Rust programming language.</i>
</p>

-----

## Overview

`differential-equations` provides a suite of numerical methods for solving differential equations, written entirely in Rust. By taking advantage of Rust's type system and generics, the library achieves higher performance than classic C/Fortran implementations. Most solvers implemented are algorithmically equivalent adaptations of classic solvers, and in other cases, are simplifications or adaptations of existing algorithms to take advantage of Rust's features.

## Table of Contents

- [Introduction](./introduction.md)
- [Quick Start](./quick_state.md)
- [Architecture](./architecture.md)
- [Defining a Differential Equation](./defining-a-differential-equation.md)
- [Setting Up a Solver](./setting-up-a-solver.md)
- [Event Handling](./event-handling.md)
- [Solout Control](./solout-control.md)
- [Solution Handling](./solution-handling.md)
- [Ordinary Differential Equations](./ode.md)
- [Delay Differential Equations](./dde.md)
- [Stochastic Differential Equations](./sde.md)

## Supported Equation Types

- Ordinary differential equations (ODEs)
- Delay differential equations (DDEs)
- Stochastic differential equations (SDEs)
... Contributions for other types of differential equations are welcome!

## Key Features

- **High Performance**: Optimized implementations that outperform traditional C/Fortran libraries
- **Type Safety**: Leverages Rust's type system for compile-time guarantees
- **Flexible State Types**: Support for custom state types with the `#[derive(State)]` attribute
- **Comprehensive Solvers**: Wide range of numerical methods from simple to sophisticated
- **Event Handling**: Built-in support for detecting and handling events during integration
- **Custom Output Control**: Fine-grained control over solution output points
- **Dense Output**: Continuous solution representation for all solvers
- **Generic Implementation**: Works with different floating-point types (`f32`/`f64`) and state representations, when `f128` is stable, it will be supported as well.

## Implemented Solvers

Some notable examples of solvers implemented in this library are:
- RK4: Runge-Kutta 4th order method
- DOP853: Dormand-Prince 8th order adaptive step method with dense output
- DOPRI5: Dormand-Prince 5th order adaptive step method with dense output
- RKV98: Verner's 9th order adaptive step method with dense output
- Radau5: Radau 5th order implicit method with dense output

More information on the solvers can be found in the respective equation type documentation (e.g., [ODE](./ode.md), [DDE](./dde.md), [SDE](./sde.md), ...).

## Motivation

This library is inspired by [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) and [SciPy](https://github.com/scipy/scipy)'s `solve_ivp` function. After switching to Rust, it was found that currently available libraries supporting differential equations shared a few common problems or missing features:

- No control over solution output (`solout`) between steps.
- Event handling was not implemented or didn't iterate to find the event point.
- Used `Box<dyn Fn>` for function pointers, which is slower than using generics.
- Used closures for defining the function to be solved, which is highly limiting.
- Only supported a few solvers.
- Dense/Continuous output was not supported.
- Used heap allocations for the state vector, which is not ideal for performance.

## Design Philosophy

This library is built on a trait-driven design philosophy, with key improvements for differential-equation solvers in the Rust ecosystem:

- **Trait-driven framework** for defining differential equations and implementing solvers
- **Standardized interfaces** that simplify the implementation of new solvers
- **Customizable output** without requiring looping a stepper
- **Event handling** with root-finding or non-terminating event handling via `solout` functions
- **Dense output support** for all solvers (or at minimum, a cubic Hermite interpolant)
- **Generic implementation** for float, state, callback, and other types

Great care was taken to maximize performance, avoid requiring users to assign generics, and create a simple idiomatic API that combines all the above features.