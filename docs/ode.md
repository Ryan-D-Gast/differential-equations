# Ordinary Differential Equations (ODE)

The `ode` module provides tools for solving ordinary differential equations (ODEs), specifically focusing on initial value problems (ODEProblems).

## Table of Contents

- [Defining a ODE](#defining-an-ode)
- [Solving an Initial Value Problem (ODEProblem)](#solving-an-initial-value-problem)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Notation](#notation)

## Defining an ODE

The `ODE` trait defines the differential equation `dydt = f(t, y)` for the solver. The differential equation is used to solve the ordinary differential equation. The trait also includes a `event` function to interrupt the solver when a condition is met or an event occurs.

### ODE Trait
* `diff` - Differential Equation `dydt = f(t, y)` in the form `f(t, &y, &mut dydt)`.
* `event` - Optional event function to interrupt the solver when a condition is met by returning `ControlFlag::Terminate(reason: CallBackData)`. The `event` function by default returns `ControlFlag::Continue` and thus is ignored. Note that `CallBackData` is by default a `String` but can be replaced with anything implementing `Clone + Debug`.

### Solout Trait
* `solout` - function to choose which points to save in the solution. This is useful when you want to save only points that meet certain criteria. Common implementations are included in the `solout` module. The `ODEProblem` trait implements methods to use them easily without direct interaction as well e.g. `even`, `dense`, and `t_eval`.

### Implementation
```rust
// Includes required elements and common methods.
// Less common methods are in the `methods` module
use differential_equations::prelude::*; 
use nalgebra::{SVector, vector};

struct LogisticGrowth {
    k: f64,
    m: f64,
}

impl ODE<f64, SVector<f64, 1>> for LogisticGrowth {
    fn diff(&self, t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0] * (1.0 - y[0] / self.m);
    }

    fn event(&self, t: f64, y: &SVector<f64, 1>) -> ControlFlag {
        if y[0] > 0.9 * self.m {
            ControlFlag::Terminate("Reached 90% of carrying capacity".to_string())
        } else {
            ControlFlag::Continue
        }
    }
}
```

Note that for clarity, the `ODE` is defined with generics `<T, V>` where `T` is the float type (e.g. `f64` or `f32`) and `V` is the state vector of the system of ordinary differential equations. By default the generics are `f64, f64` and thus can be omitted if the system is a single ODE with a `f64` type and a single state variable `f64`. Here a `SVector` is used despite `f64` being usable here for clarity. For example a system with multiple ODEs of size N then `SVector<f64, N>` can be used.

## Solving an Initial Value Problem

The `ODEProblem` trait is used to solve the system using the solver. The trait includes methods to set the initial conditions, solve the system, and get the solution. The `solve` method returns a `Result<Solution, Status>` where `Solution` is a struct containing the solution including fields with outputted t, y, and the solver status, and `Status` is returned with the error if it occurs. In addition, statistics including steps, evals, rejected steps, accepted steps, and the solve time are included as fields in the `Solution` struct.

```rust
fn main() {
    let mut method = ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12);
    let y0 = vector![1.0];
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_problem = ODEProblem::new(ode, t0, tf, y0);
    match logistic_growth_problem
        .even(1.0)          // uses EvenSolout to save with dt of 1.0
        .solve(&mut method) // Solve the system and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let Status::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("Solver stopped: {}", reason);
            }

            // Print the solution
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("({:.4}, {:.4})", t, y[0]);
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {}", solution.steps);
            println!("Rejected Steps: {}", solution.rejected_steps);
            println!("Accepted Steps: {}", solution.accepted_steps);
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
```

### Output

```sh
Solver stopped: Reached 90% of carrying capacity
Solution:
(0.0000, 1.0000)
(1.0000, 2.3197)
(2.0000, 4.5085)
(3.0000, 6.9057)
(4.0000, 8.5849)
(4.3944, 9.0000)
Function evaluations: 325
Steps: 22
Rejected Steps: 2        
Accepted Steps: 20
```

## Examples

For more examples, see the `examples` directory. The examples demonstrate different systems, methods, and output methods for different use cases.

| Example | Description & Demonstrated Features |
|---|---|
| [Exponential Growth](../../examples/ode/01_exponential_growth/main.rs) | Solves a simple exponential growth equation using the `dop853` method. Demonstrates basic usage of `ODEProblem` and `ODE` traits. Manually prints results from `Solution` struct fields. |
| [Harmonic Oscillator](../../examples/ode/02_harmonic_oscillator/main.rs) | Simulates a harmonic oscillator system using `rk4` method. Uses a condensed setup to demonstrate chaining to solve without intermediate variables. Uses `last` method on solution to conveniently get results and print. |
| [Logistic Growth](../../examples/ode/03_logistic_growth/main.rs) | Models logistic growth with a carrying capacity. Demonstrates the use of the `event` function to stop the solver based on a condition. In addition shows the use of `even` output for `ODEProblem` setup and `iter` method on the solution for output. |
| [SIR Model](../../examples/ode/04_sir_model/main.rs) | Simulates the SIR model for infectious diseases. Uses the `AdamsPredictorCorrector::v4()` method to solve the system. Uses struct as the `State` via `derive(State)` and a custom event termination enum. |
| [Damped Pendulum](../../examples/ode/05_damped_pendulum/main.rs) | Simulates a simple pendulum using the `rkf45` solver. Shows the use of `problem.t_eval()` to define specific points to be saved e.g. `t_eval(vec![1.0, 3.0, 7.5, 10.0])` |
| [Integration](../../examples/ode/06_integration/main.rs) | Demonstrates the differences between `even`, `dense`, `t_eval`, and the default solout methods for a simple differential equation with an easily found analytical solution. |
| [Cr3bp](../../examples/ode/07_cr3bp/main.rs) | Simulates the Circular Restricted Three-Body Problem (CR3BP) using the `dop853` method. Uses the `hyperplane_crossing` method to log when the spacecraft crosses a 3D plane. |
| [Damped Oscillator](../../examples/ode/08_damped_oscillator/main.rs) | Demonstrates the use of the `crossing` method to use the CrossingSolout to log instances where a crossing occurs. In this case, the example saves points where the position is at zero. |
| [Matrix ODE](../../examples/ode/09_matrix_ode/main.rs) | Solves a system of ODEs using a matrix system. Demonstrates how to define a system of equations using matrices. |
| [Custom Solout](../../examples/ode/10_custom_solout/main.rs) | Demonstrates how to create a custom `Solout` implementation to save points based on a custom condition. In addition inside the Solout struct additional calculations are stored each step and accessible after solving is complete. |
| [Schrodinger](../../examples/ode/11_schrodinger/main.rs) | Solves the time-dependent Schrödinger equation using the `dop853` method. Demonstrates the use of complex numbers in the ODE system. |
| [Brusselator](../../examples/ode/12_brusselator/main.rs) | Demonstrates solving a stiff system using implicit Runge-Kutta methods (`gauss_legendre_6`) with analytical Jacobian provided to accelerate Newton iterations. |

## Benchmarks

Test results from [differential-equations-comparison](https://github.com/Ryan-D-Gast/differential-equations-comparison) github repository which compares the performance of the `differential-equations` library with Fortran implementations of the DOP853 method.

| Problem | Implementation | Mean [ms] | Min [ms] | Max [ms] | Relative |
| :---: |:---:|---:|---:|---:|---:|
| Van der Pol Osc. | Rust | 8.8 ± 0.3 | 8.2 | 10.3 | 1.00 |
| Van der Pol Osc. | Fortran | 12.5 ± 2.9 | 11.5 | 75.1 | 1.41 ± 0.33 |
| CR3BP | Rust | 7.0 ± 0.4 | 6.5 | 8.6 | 1.00 |
| CR3BP | Fortran | 9.8 ± 1.8 | 9.1 | 46.0 | 1.40 ± 0.26 |

Testing has shown that the `differential-equations` Rust implementation is at about 10% to 40% faster than the Fortran implementations of the DOP853 method.

## Notation

Typical ODE libraries either use `x` or `t` for the independent variable and `y` for the dependent variable. This library uses the following notation:
- `t` - The independent variable, typically time often `x` in other ode libraries.
- `y` - The dependent variable, instead of `x` to avoid confusion with an independent variable in other notations.
- `dydt` - The derivative of `y` with respect to `t`.
- `k` - The coefficients of the solver, typically a derivative such as in the Runge-Kutta methods.

Any future methods added to the library should follow this notation to maintain consistency.
