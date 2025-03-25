<p align="center">
  <img src="./assets/logo.svg" width="100%" max-width="800px" alt="differential-equations">
</p>

[![Crates.io](https://img.shields.io/crates/v/differential-equations.svg)](https://crates.io/crates/differential-equations) [![Documentation](https://docs.rs/differential-equations/badge.svg)](https://docs.rs/differential-equations)

`differential-equations` is a Rust library for solving various differential equations. The library currently focuses on ordinary differential equations (ODEs) with planned support for other types of differential equations in the future.

## Features

- **Ordinary Differential Equations (ODE)**: Solve initial value problems with various numerical methods
  - Multiple fixed-step and adaptive-step solvers
  - Event detection and handling
  - Customizable output control
  - High performance implementation

## Documentation

For detailed documentation on each module:

- [Overview](./docs/README.md)
- [Ordinary Differential Equations (ODE)](./docs/ode/introduction.md)

## Example

```rust
use differential_equations::ode::*;

// Define a simple exponential growth model
struct ExponentialGrowth {
    rate: f64,
}

impl System for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.rate * y[0];
    }
}

fn main() {
    let system = ExponentialGrowth { rate: 0.1 };
    let t0 = 0.0;
    let tf = 10.0;
    let y0 = vector![1.0];

    let solver = DOP853::new().rtol(1e-6).atol(1e-6);
    
    // Create and solve the IVP
    match IVP::new(system, t0, tf, y0)
        .even(1.0)  // Save solution at regular intervals
        .solve(&mut solver)
    {
        Ok(sol) => {
            for (t, y) in sol.iter() {
                println!("t = {:.1}, y = {:.6}", t, y[0]);
            }
        }
        Err(e) => panic!("Failed to solve the IVP: {}", e),
    };
}
```

## Installation

To use `differential-equations` in your Rust project, add it as a dependency using `cargo`:

```sh
cargo add differential-equations
```

## Citation

If you use this library in your research, please consider citing it as follows:

```bibtex
@software{differential-equations,
  author = {Ryan D. Gast},
  title = {differential-equations: A Rust library for solving differential equations.},
  url = {https://github.com/Ryan-D-Gast/differential-equations},
  version = {0.1.0},
}
```

## References

The following references were used in the development of this library:

1. Burden, R.L. and Faires, J.D. (2010) [Numerical Analysis. 9th Edition](https://dl.icdst.org/pdfs/files3/17d673b47aa520f748534f6292f46a2b.pdf), Brooks/Cole, Cengage Learning, Boston.
2. E. Hairer, S.P. Norsett and G. Wanner, "[Solving ordinary Differential Equations I. Nonstiff Problems](http://www.unige.ch/~hairer/books.html)", 2nd edition. Springer Series in Computational Mathematics, Springer-Verlag (1993).
3. Ernst Hairer's website: [Fortran and Matlab Codes](http://www.unige.ch/~hairer/software.html)
