# Boundary Value Problems (BVP)

Boundary value problems solve differential equations with constraints at both
ends of the interval. The `BVP` builder mirrors the `IVP` builder style, but it
uses boundary residuals instead of fixed initial conditions.

For ODE BVPs, define the differential equation with the regular `ODE` trait and
define the endpoint residual with `Boundary`. The same equation type can be used
for IVP solves with `IVP::ode` and BVP solves with `BVP::ode` when it implements
both traits.

## Defining an ODE BVP

The `Boundary` trait receives the state at the left endpoint (`y_a`), the state
at the right endpoint (`y_b`), and writes a residual into `res`. A BVP is solved
when every residual component is zero.

```rust
use differential_equations::prelude::*;

struct HarmonicOscillator;

impl ODE<f64, [f64; 2]> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

impl Boundary<f64, [f64; 2]> for HarmonicOscillator {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        res[0] = y_a[0];
        res[1] = y_b[0] - 1.0;
    }
}

let problem = HarmonicOscillator;
let method = Shooting::single(ExplicitRungeKutta::dop853());
let solution = BVP::ode(
    &problem,
    0.0,
    std::f64::consts::FRAC_PI_2,
    [0.0, 0.5],
)
.method(method)
.solve()?;
# Ok::<(), differential_equations::error::Error<f64, [f64; 2]>>(())
```

## Closure-Based Setup

For closure-based setup, use `BVP::ode_from_fn`:

```rust
use differential_equations::prelude::*;

let method = Shooting::single(ExplicitRungeKutta::dop853());
let solution = BVP::ode_from_fn(
    |_t, y: &[f64; 2], dydt: &mut [f64; 2]| {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    },
    |y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]| {
        res[0] = y_a[0];
        res[1] = y_b[0] - 1.0;
    },
    0.0,
    std::f64::consts::FRAC_PI_2,
    [0.0, 0.5],
)
.method(method)
.solve()?;
# Ok::<(), differential_equations::error::Error<f64, [f64; 2]>>(())
```

## Pipe Heat Transfer Example

The ODE examples include a steady pipe heat-transfer BVP:

```text
T'' = beta^2 (T - T_ambient)
T(0) = T_inlet
dT/dx(L) = 0
```

The first-order state is `[T, dT/dx]`; `Boundary` fixes the inlet temperature and
the insulated outlet gradient while `Shooting` finds the initial temperature
gradient.
See [Pipe Heat Transfer](../examples/ode/18_pipe_heat_transfer/main.rs).

## Shooting Methods

`Shooting::single` solves ODE BVPs by repeatedly solving one IVP across the full
interval and applying Newton iteration to the guessed initial state.

```rust
let method = Shooting::single(ExplicitRungeKutta::dop853())
    .tolerance(1e-8)
    .max_iterations(50);
```

`Shooting::multiple` partitions the interval into subintervals, solves an IVP on
each segment, and uses Newton iteration to satisfy both the endpoint boundary
conditions and continuity between neighboring segment states.

```rust
let method = Shooting::multiple(ExplicitRungeKutta::dop853())
    .segments(5)
    .tolerance(1e-8)
    .max_iterations(50);
```

For both methods, the boundary residual state must have the same dimension as
the ODE state.

## Output and Statistics

`BVP` supports the same output controls as `IVP` for the final converged ODE
trajectory:

```rust
let solution = BVP::ode(&problem, 0.0, 1.0, [0.0, 0.0])
    .t_eval([0.0, 0.25, 0.5, 0.75, 1.0])
    .method(Shooting::single(
        ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10),
    ))
    .solve()?;
# Ok::<(), differential_equations::error::Error<f64, [f64; 2]>>(())
```

Available output options include `solout`, `even`, `dense`, `t_eval`, `event`,
`crossing`, and `hyperplane_crossing`. For shooting methods, these output
controls are applied after the boundary residual has converged, so the returned
points describe the final trajectory. The returned `Solution` also includes the
function evaluations, finite-difference Jacobian evaluations, Newton iterations,
linear decompositions, solves, accepted/rejected IVP steps, and elapsed wall time
used by the shooting solve.

## API Summary

| Item | Purpose |
|---|---|
| `BVP::ode` | Build an ODE boundary value problem from a type implementing `ODE + Boundary`. |
| `BVP::ode_from_fn` | Build an ODE boundary value problem from derivative and boundary closures. |
| `Boundary` | Define the endpoint residual for a BVP. |
| `Shooting::single` | Solve ODE BVPs by iterating on the unknown initial state. |
| `Shooting::multiple` | Solve ODE BVPs by iterating on segment node states and continuity residuals. |
| `BVP::t_eval`, `BVP::even`, `BVP::dense` | Select output points for the final converged trajectory. |
