<p align="center">
  <img src="./assets/logo.svg" width="1000" alt="differential-equations">
</p>

<p align="center">
    <a href="https://crates.io/crates/differential-equations">
        <img src="https://img.shields.io/crates/v/differential-equations.svg?style=flat-square" alt="crates.io">
    </a>
    <a href="https://docs.rs/differential-equations">
        <img src="https://docs.rs/differential-equations/badge.svg" alt="docs.rs">
    </a>
    <a href="https://github.com/Ryan-D-Gast/differential-equations/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>

<p align="center">
    <strong>
        <a href="./docs/README.md">Documentation</a> |
        <a href="./examples/ode/01_exponential_growth/main.rs">Examples</a> |
        <a href="https://github.com/Ryan-D-Gast/differential-equations"
        >GitHub</a> |
        <a href="https://docs.rs/differential-equations/latest/differential_equations/">Docs.rs</a> |
        <a href="https://crates.io/crates/differential-equations">Crates.io</a>
    </strong>
</p>

-----

<p align="center">
<b>A high-performance library for numerically solving differential equations</b><br>
<i>for the Rust programming language.</i>
</p>

-----

A high-performance library for solving differential equations in Rust, including:

- **[Ordinary Differential Equations (ODEs)](./docs/ode/introduction.md)** - Fixed-step and adaptive [solvers](./docs/ode/introduction.md/#solvers) with comprehensive features including event detection, dense output, and customizable and common recipes for solution output.
    - **Initial Value Problems (IVPs)** - Solve problems with known initial conditions

- **[Stochastic Differential Equations (SDEs)](./docs/sde/introduction.md)** - Includes solvers such as Euler-Maruyama, Milstein, and Runge-Kutta methods for stochastic differential equations.

## Contributing

This library is looking for contributions to bring the future of scientific computing to Rust!

Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for more information on how to contribute to this project.
