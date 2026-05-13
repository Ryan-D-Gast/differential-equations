# Changelog

## v0.6.0 - First-Class Problem Builders, Events, and BVPs

This release is a broad API and capability release for `differential-equations`.
It expands the crate from a collection of equation-specific solver entry points
into a more consistent builder-driven interface for IVPs and BVPs, with shared
output control, event handling, statistics, and examples across equation types.

The headline feature is first-class ODE boundary value problem support, but this
release is larger than BVPs alone. It also includes major IVP ergonomics, closure
constructors, unified event handling, improved state support, stronger errors,
new methods, and test organization work.

### Highlights

- Added the unified `IVP` builder API for ODE, DAE, DDE, and SDE initial value
  problems.
- Added closure-based IVP constructors:
  - `IVP::ode_from_fn`
  - `IVP::dae_from_fn`
  - `IVP::dde_from_fn`
  - `IVP::sde_from_fn`
- Added first-class ODE BVP support through `BVP::ode` and `BVP::ode_from_fn`.
- Added the `Boundary` trait for endpoint residuals.
- Added BVP shooting methods:
  - `Shooting::single`
  - `Shooting::multiple`
- Added IVP-style output controls to BVP solves, including `solout`, `even`,
  `dense`, `t_eval`, event wrapping, component crossings, and hyperplane
  crossings.
- Added a unified event system based on an `Event` trait and `EventConfig`,
  available across IVP problem types.
- Added higher-order SDE support with `Milstein`.
- Added adaptive BDF support under the public
  `BackwardDifferentiationFormula` name.
- Added scalar `State` support for `f32` and `f64`.
- Improved `State` derive support, including non-generic state structs.
- Relaxed many bounds with `?Sized` so trait-object and `dyn` workflows are
  easier to use.
- Added a pipe heat-transfer BVP example with analytical validation and an SVG
  plot.
- Added a dedicated BVP integration test suite.

### IVP API Improvements

The IVP workflow is now centered on the `IVP` builder:

```rust
let solution = IVP::ode(&system, t0, tf, y0)
    .t_eval([0.0, 0.5, 1.0])
    .event(&event_detector)
    .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10))
    .solve()?;
```

The same builder pattern is available for other equation categories:

```rust
let ode = IVP::ode(&system, t0, tf, y0);
let dae = IVP::dae(&system, t0, tf, y0);
let dde = IVP::dde(&system, t0, tf, y0, history);
let sde = IVP::sde(&mut system, t0, tf, y0);
```

Closure-backed constructors make lightweight models easier to express without
declaring a dedicated struct:

```rust
let solution = IVP::ode_from_fn(
    |_t, y: &f64, dydt: &mut f64| *dydt = *y,
    0.0,
    1.0,
    1.0,
)
.method(ExplicitRungeKutta::dop853())
.solve()?;
```

This release also standardizes solver output controls through the builder. Users
can choose default step output, evenly spaced output, dense output, explicit
evaluation points, crossing output, hyperplane crossing output, or a custom
`Solout` implementation.

### Event Handling

Event handling has been redesigned around a separate event detector object:

```rust
impl Event<f64, f64> for PopulationThreshold {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Positive, Some(1))
    }

    fn event(&self, _t: f64, y: &f64) -> f64 {
        *y - self.threshold
    }
}
```

Events compose with output controls:

```rust
let solution = IVP::ode(&system, t0, tf, y0)
    .dense(5)
    .event(&threshold)
    .method(ExplicitRungeKutta::dop853())
    .solve()?;
```

This gives ODE, DAE, DDE, and SDE workflows a shared, SciPy-like event model with
direction filtering and termination counts.

### Boundary Value Problems

ODE BVPs now have their own public builder:

```rust
let method = Shooting::single(
    ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10),
);

let solution = BVP::ode(&problem, t0, tf, initial_guess)
    .even(0.01)
    .method(method)
    .solve()?;
```

BVPs reuse the regular `ODE` trait for the differential equation and add a
separate `Boundary` trait for endpoint residuals:

```rust
impl Boundary<f64, [f64; 2]> for HarmonicOscillator {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        res[0] = y_a[0];
        res[1] = y_b[0] - 1.0;
    }
}
```

Two shooting methods are included:

- `Shooting::single` adjusts one initial state and solves one IVP across the
  full interval per Newton iteration.
- `Shooting::multiple` partitions the interval, solves IVPs on each segment, and
  enforces continuity between segment node states.

BVP results use the same `Solution` type as IVPs and report accumulated
statistics for internal IVP solves, finite-difference Jacobian evaluations,
Newton iterations, LU decompositions, and linear solves.

### New Examples and Tests

This release adds a steady pipe heat-transfer BVP example:

```text
T'' = beta^2 (T - T_ambient)
T(0) = T_inlet
dT/dx(L) = 0
```

The example lives at:

```text
examples/ode/18_pipe_heat_transfer/main.rs
```

It validates the numerical BVP result against the analytical insulated-end
profile and writes:

```text
examples/ode/18_pipe_heat_transfer/pipe_heat_transfer.svg
```

The new BVP test suite covers:

- single shooting on a harmonic oscillator BVP,
- multiple shooting on a harmonic oscillator BVP,
- BVP output control with `even`,
- trait-object BVP solving with `dyn`,
- single and multiple shooting on the pipe heat-transfer BVP.

### Naming and Organization

Several names and modules were cleaned up for long-term consistency:

- The high-level initial value problem builder is `IVP`.
- The new boundary value problem builder is `BVP`.
- Equation-specific solve files are named `solve_ivp.rs`.
- The low-level ODE BVP entry point is `solve_bvp.rs`.
- BVP methods live under `src/methods/bvp`.
- The public BDF method name is `BackwardDifferentiationFormula`.

The intended shape is now:

```rust
IVP::ode(...).method(...).solve()
BVP::ode(...).method(...).solve()
```

That leaves room for future equation categories and BVP methods without creating
one-off folder structures or special-case APIs.

### Error Handling and State Support

Errors now provide more actionable messages for common solver failure modes,
including bad inputs, maximum step count, tiny step size, stiffness,
interpolation bounds, DDE history issues, and linear algebra failures.

State support has also expanded:

- `f32` and `f64` can be used directly as scalar states.
- The `State` derive macro supports non-generic structs.
- More APIs accept `?Sized` equation types, improving compatibility with trait
  objects and dynamic dispatch.

### Upgrade Notes

This is a pre-1.0 crate, so some API cleanup is expected. The main things to
watch when upgrading are:

- Use `IVP`, not older initial-value builder names.
- Use `BackwardDifferentiationFormula`, not the shorter BDF public name.
- Configure tolerances on the solver method, not on the problem builder:

```rust
let method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
let solution = IVP::ode(&system, t0, tf, y0).method(method).solve()?;
```

- Use the separate `Event` trait for event detection instead of embedding event
  behavior in equation traits.
- For BVPs, implement `ODE` for the differential equation and `Boundary` for the
  endpoint residual.

### Suggested Release Announcement

`differential-equations` v0.6.0 is a major API and capability release focused on
making the crate more consistent, more extensible, and easier to use for real
scientific computing workflows.

The biggest new capability is first-class ODE boundary value problem support
with `BVP::ode`, `Boundary`, `Shooting::single`, and `Shooting::multiple`. BVPs
now feel like IVPs: choose a problem, choose output, choose a method, and call
`solve`.

At the same time, IVP workflows have been cleaned up and expanded. ODE, DAE,
DDE, and SDE solves now share the `IVP` builder style, closure-based constructors
are available for quick problem definitions, event handling is unified, and
output control is consistently available through builder methods.

This release also adds higher-order SDE support, adaptive BDF, scalar state
support, stronger errors, improved `State` derive behavior, trait-object
compatibility work, more examples, and a new BVP test suite.

The result is a more coherent foundation for future solvers and equation types.
