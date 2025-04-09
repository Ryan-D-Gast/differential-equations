# Ordinary Differential Equations (ODE)

The `ode` module provides tools for solving ordinary differential equations (ODEs), specifically focusing on initial value problems (IVPs).

## Table of Contents

- [Solvers](#solvers)
- [Defining a ODE](#defining-a-system)
- [Solving an Initial Value Problem (IVP)](#solving-an-initial-value-problem-ivp)
- [Examples](#examples)
- [Notation](#notation)
- [Benchmarks](#benchmarks)

## Solvers

The module includes a set of solvers for solving ODEs. The solver algorithmic core and coefficients are implemented as structs implementing the `Solver` trait. The solver's settings can then be configured before being used in the `ivp.solve(&mut solver)` method which acts as the controller for the solver.

### Fixed Step Size

| Solver | Description |
|--------|-------------|
| `Euler` | Euler's method (1st order Runge-Kutta) |
| `Midpoint` | Midpoint method (2nd order Runge-Kutta) |
| `Heuns` | Heun's method (2nd order Runge-Kutta) |
| `Ralston` | Ralston's method (2nd order Runge-Kutta) |
| `RK4` | Classical 4th order Runge-Kutta method |
| `ThreeEights` | 3/8 Rule 4th order Runge-Kutta method |
| `APCF4` | Adams-Predictor-Corrector 4th order fixed step-size method |

### Adaptive Step Size

| Solver | Description |
|--------|-------------|
| `RKF` | Runge-Kutta-Fehlberg 4(5) adaptive method |
| `CashKarp` | Cash-Karp 4(5) adaptive method |
| `DOPRI5` | Dormand-Prince 5(4) adaptive method with dense output |
| `DOP853` | Dormand-Prince 8(5,3) adaptive method with dense output |
| `RKV65` | Verner's 6(5) method with dense output |
| `RKV98` | Verner's 9(8) method with dense output |
| `APCV4` | Adams-Predictor-Corrector 4th order variable step-size method |

**Note:** `dense output` references the solver having an high-order interpolant to provide accurate results between `t_prev` and `t` between steps. Often these require a few extra function evaluations. The order of the interpolant can be found in the docs of the solver. The remaining solvers use a cubic Hermite polynomial to provide interpolated results. For event detection the interpolant for each solver is used to find the location of the event. 

## Solvers Comparison

Solver Selection Tables below provide detailed rankings of all available solvers across key metrics to help you select the right one for your specific needs.

### 1. Accuracy Ranking

| Rank | Solver | Order | Accuracy Level |
|------|--------|-------|----------------|
| 1 | RKV98 | 9(8) | Very High |
| 2 | DOP853 | 8(5,3) | High |
| 3 | RKV65 | 6(5) | Medium-High |
| 4 | DOPRI5 | 5(4) | Medium |
| 5 | RKF, CashKarp | 4(5) | Medium-Low |
| 6 | RK4, ThreeEights, APCF4, APCV4 | 4 | Low |
| 7 | Midpoint, Heuns, Ralston | 2 | Very Low |
| 8 | Euler | 1 | Minimal |

> Note: Higher-order methods are note just more accurate but often quicker as they can take larger steps. The sweet spot for most problems is between 4th and 8th order methods. The `DOPRI5` and `DOP853` solvers are recommended for most general-purpose applications due to their balance of accuracy and efficiency. The `RKV98` solver is the best choice for high-precision requirements where a large part of the error is due to floating-point error.

### 2. Memory Usage Ranking

| Rank | Solver | Stages | Memory Requirement |
|------|--------|--------|-------------------|
| 1 | Euler | 1 | Minimal |
| 2 | Midpoint, Heuns, Ralston | 2 | Very Low |
| 3 | RK4, ThreeEights | 4 | Low |
| 4 | RKF, CashKarp | 6 | Medium-Low |
| 5 | DOPRI5 | 7 | Medium |
| 6 | RKV65 | 8 | Medium-High |
| 7 | DOP853 | 12 | High |
| 8 | RKV98, APCF4, APCV4 | 16+ | Very High |

### 3. Best Use Cases

| Problem Type | Recommended Solvers | Reasoning |
|--------------|---------------------|-----------|
| Quick prototyping | RK4 | Simple, reliable, good balance |
| Non-stiff, low precision | RKF, CashKarp | Efficient with adequate accuracy |
| General purpose | DOPRI5 | Good balance of accuracy and speed |
| High accuracy needs | DOP853, RKV65 | Excellent precision with reasonable efficiency |
| Extremely high precision | RKV98 | Highest accuracy available |

### 4. Adaptive vs Fixed Trade-offs

| Feature | Adaptive Solvers | Fixed-Step Solvers |
|---------|------------------|-------------------|
| Step size control | Automatic based on error tolerance | Manual specification required |
| Efficiency | High | Low |
| Predictable computation time | No | Yes |
| Memory overhead | Higher | Lower |
| Best for | Problems with varying dynamics | Predictable, smooth problems |
| Examples | DOPRI5, DOP853, RKV65 | RK4, Euler |

> Note: The main trade-off between using an adaptive or fixed-step method is the cost of the differential equation. If it is a complex function then reducing the number of evaluations via an adaptive step size method is optimal. If the function is very simple, relatively short distance between `t0` and `tf` then fixed step size could be optimal if the high memory usage of the adaptive method is a concern. Given all the stipulations for using fixed step size methods for almost all problems, it is recommended to use adaptive step size methods.

## Defining a ODE

The `ODE` trait defines the differential equation `dydt = f(t, y)` for the solver. The differential equation is used to solve the ordinary differential equation. The trait also includes a `event` function to interrupt the solver when a condition is met or an event occurs.

### ODE Trait
* `diff` - Differential Equation `dydt = f(t, y)` in the form `f(t, &y, &mut dydt)`.
* `event` - Optional event function to interrupt the solver when a condition is met by returning `ControlFlag::Terminate(reason: CallBackData)`. The `event` function by default returns `ControlFlag::Continue` and thus is ignored. Note that `CallBackData` is by default a `String` but can be replaced with anything implementing `Clone + Debug`.

### Solout Trait
* `solout` - function to choose which points to save in the solution. This is useful when you want to save only points that meet certain criteria. Common implementations are included in the `solout` module. The `IVP` trait implements methods to use them easily without direct interaction as well e.g. `even`, `dense`, and `t_eval`.

### Implementation
```rust
// Includes required elements and common solvers.
// Less common solvers are in the `solvers` module
use differential_equations::ode::*; 
use nalgebra::{SVector, vector};

struct LogisticGrowth {
    k: f64,
    m: f64,
}

impl ODE<f64, 1, 1> for LogisticGrowth {
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

Note that for clarity, the `ODE` is defined with generics `<T, R, C>` where `T` is the float type and `R, C` is the dimension of the system of ordinary differential equations. By default the generics are `f64, 1, 1` and thus can be omitted if the system is a single ODE with a `f64` type.

## Solving an Initial Value Problem (IVP)

The `IVP` trait is used to solve the system using the solver. The trait includes methods to set the initial conditions, solve the system, and get the solution. The `solve` method returns a `Result<Solution, SolverStatus>` where `Solution` is a struct containing the solution including fields with outputted t, y, and the solver status, and `SolverStatus` is returned with the error if it occurs. In addition, statistics including steps, evals, rejected steps, accepted steps, and the solve time are included as fields in the `Solution` struct.

```rust
fn main() {
    let mut solver = DOP853::new().rtol(1e-12).atol(1e-12);
    let y0 = vector![1.0];
    let t0 = 0.0;
    let tf = 10.0;
    let ode = LogisticGrowth { k: 1.0, m: 10.0 };
    let logistic_growth_ivp = IVP::new(ode, t0, tf, y0);
    match logistic_growth_ivp
        .even(1.0)          // uses EvenSolout to save with dt of 1.0
        .solve(&mut solver) // Solve the system and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let SolverStatus::Interrupted(ref reason) = solution.status {
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
Function evaluations: 359
Steps: 25
Rejected Steps: 4        
Accepted Steps: 21
```

## Examples

For more examples, see the `examples` directory. The examples demonstrate different systems, solvers, and output methods for different use cases.

| Example | Description & Demonstrated Features |
|---|---|
| [Exponential Growth](../../examples/ode/01_exponential_growth/main.rs) | Solves a simple exponential growth equation using the `DOP853` solver. Demonstrates basic usage of `IVP` and `ODE` traits. Manually prints results from `Solution` struct fields. |
| [Harmonic Oscillator](../../examples/ode/02_harmonic_oscillator/main.rs) | Simulates a harmonic oscillator system using `RK4` method. Uses a condensed setup to demonstrate chaining to solve without intermediate variables. Uses `last` method on solution to conveniently get results and print. |
| [Logistic Growth](../../examples/ode/03_logistic_growth/main.rs) | Models logistic growth with a carrying capacity. Demonstrates the use of the `event` function to stop the solver based on a condition. In addition shows the use of `even` output for `IVP` setup and `iter` method on the solution for output. |
| [SIR Model](../../examples/ode/04_sir_model/main.rs) | Simulates the SIR model for infectious diseases. Uses the `APCV4` solver to solve the system. Uses custom event termination enum. |
| [Damped Pendulum](../../examples/ode/05_damped_pendulum/main.rs) | Simulates a simple pendulum using the `RKF` solver. Shows the use of `ivp.t_out` to define points to be saved e.g. `t_out = [1.0, 3.0, 7.5, 10.0]` |
| [Integration](../../examples/ode/06_integration/main.rs) | Demonstrates the differences between `even`, `dense`, `t_out`, and the default solout methods for a simple differential equation with an easily found analytical solution. |
| [Cr3bp](../../examples/ode/07_cr3bp/main.rs) | Simulates the Circular Restricted Three-Body Problem (CR3BP) using the `DOP853` solver. Uses the `hyperplane_crossing` method to log when the spacecraft crosses a 3D plane. |
| [Damped Oscillator](../../examples/ode/08_damped_oscillator/main.rs) | Demonstrates the use of the `crossing` method to use the CrossingSolout to log instances where a crossing occurs. In this case, the example saves points where the position is at zero. |
| [Matrix ODE](../../examples/ode/09_matrix_ode/main.rs) | Solves a system of ODEs using a matrix system. Demonstrates how to define a system of equations using matrices. |
| [Custom Solout](../../examples/ode/10_custom_solout/main.rs) | Demonstrates how to create a custom `Solout` implementation to save points based on a custom condition. In addition inside the Solout struct additional calculations are stored and then accessed via `Solution.solout.(fields)` |

## Benchmarks

Included in the [ode comparion folder](../../comparison/ode/README.md) which contains benchmarks comparing the speed of the `DOP853` solver implementation in `differential-equations` against implementations in other programming languages including Fortran.

A sample result via `comparison/ode/...` is shown below:

[![Benchmark Results](./ode_benchmark.png)](../differential-equations-benchmarks "Averaged over 100 runs for each problem per solver implementation")

Testing has shown that the `differential-equations` Rust implementation is about 10% faster than the Fortran implementations above. Take the result with a grain of salt as more testing by other users is needed to confirm the results.

## Notation

Typical ODE libraries either use `x` or `t` for the independent variable and `y` for the dependent variable. This library uses the following notation:
- `t` - The independent variable, typically time often `x` in other ode libraries.
- `y` - The dependent variable, instead of `x` to avoid confusion with an independent variable in other notations.
- `dydt` - The derivative of `y` with respect to `t`.
- `k` - The coefficients of the solver, typically a derivative such as in the Runge-Kutta methods.

Any future solvers added to the library should follow this notation to maintain consistency.
