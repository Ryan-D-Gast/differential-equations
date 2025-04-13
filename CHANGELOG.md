# Changelog

**Legend:** `.` Minor change · `-` Bug fix · `+` New feature · `^` Improvement · `!` Breaking change · `*` Refactor

## 2025-04-12 - `0.1.3`
- `^` Refactored module imports in codebase with changes in a cleaner easier to read structure.
- `!` `SolverError` and `SolverStatus` renamed to `Error` and `Status respectively. In addition moved to root of the crate as these are planed to be shared for all differential equation types.
- `!` Changed `Solver` trait to `NumericalMethod` as it is more descriptive and less vague.
- `.` Took Error variants out of `SolverStatus` and put them in their own enum `SolverError`. If the user was previous handling specific errors, this is breaking change but otherwise no code changes are needed.
- `!` Removed `SolutionInterface` trait and have Solout function take a mutable reference to the `Solution` struct instead. Methods which are dangerous to call during Solout have been removed or made only public to crate to prevent misuse.
- `.` Remove `Solout.include_t0_tf()` which defaulted to `true` and instead Solout is called called before first step. Common default implementations required minor modifies to prevent NaN issues.
- `!` Changed `EventAction` and `EventData` into `ControlFlag` and `CallBackData` respectively. In addition, `Solout` now returns `ControlFlag`. Terminations for `Solout` are not iterated over to find the expact point instead just terminate immediately.
- `^` Removed nalgebra re-export requiring users to add it as a dependency and manage the version themselves.

## 2025-04-05 - `0.1.2`
- `+` Added Verner Runge-Kutta methods with dense output support: `RKV65` and `RKV98`
- `^` Enhanced `DOPRI5` method with optimized implementation matching Fortran versions, reducing memory usage and adding dense output support
- `!` Changed `Solout` function signature to `fn(&mut Solver, &mut SolutionInterface)`, providing direct access to solution storage. See [custom solout](./examples/ode/10_custom_solout/main.rs) for an example of usage.
- `!` Modified `Solver.interpolate(t)` to return a `Result<_, InterpolationError>` instead of hoping it was called within the interpolation range `t_prev..=t`.
- `*` Relocated step tracking (steps, accepted, rejected) from `Solver` to the `Solution` struct updated in `solve_ivp`.
- `-` Fixed bug where `rejected_steps` was not being counted in the total `steps` in some solvers.

## 2025-03-30 - `0.1.1`

- `!` Renamed `System` trait to `ODE` for consistency with future differential equation types (e.g., `PDE`, `SDE`)
- `*` Refactored documentation structure with modular approach to accommodate future equation types
- `^` Improved README.md with clearer module navigation and installation instructions

## 2025-03-25 - `0.1.0`

- `+` Initial release of differential-equations
- `+` Stable version of ODE solvers migrated from `rgode`
Note changes will be made to ODE section such as adding more solvers, but API is expected to remain stable.
    - `+` Fixed-step solvers: Euler, Midpoint, Heun's, Ralston, RK4, ThreeEights, APCF4
    - `+` Adaptive-step solvers: RKF, CashKarp, DOPRI5, DOP853, APCV4
    - `+` System trait for defining ODE systems
    - `+` Custom solout functionality
    - `+` Event detection
    - `+` Flexible IVP solver with dense output, event detection, custom output control and more.
    - `+` Comprehensive documentation and examples
    - `+` Optional Polars integration for data export