# Changelog

**Legend:** `.` Minor change · `-` Bug fix · `+` New feature · `^` Improvement · `!` Breaking change · `*` Refactor

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