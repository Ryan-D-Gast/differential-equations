# Feature Roadmap

This roadmap outlines the implementation plan for reaching feature parity with advanced differential equation solver suites (such as `diffsol`, `diffrax`, and `DifferentialEquations.jl`).

## Current Status & Parity Analysis

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Linear Algebra** | 🟢 Partial | Supports `nalgebra` and internal `Matrix`. Missing `faer` and sparse support. |
| **Adaptive Step-size** | 🟢 Supported | Implemented via `Tolerance` enum. Currently unified for the whole state. |
| **Dense Output** | 🟢 Supported | Implemented via `Interpolation` trait and `DenseSolout`. |
| **Event Handling** | 🟢 Supported | Supported via `Event` trait and `EventSolout`. |
| **Numerical Quadrature** | 🔴 Planned | Not yet implemented as a first-class feature. |
| **Forward Sensitivity** | 🔴 Planned | Can be done manually with dual numbers, but lacks dedicated API. |
| **Adjoint Sensitivity** | 🟡 Designing | Design proposal created in `docs/adjoint_design.md`. |

---

## Roadmap

### Phase 1: Enhanced Linear Algebra & Performance
- [ ] **`faer` Integration**: Add support for the `faer` crate as a high-performance alternative to `nalgebra`.
- [ ] **Sparse Matrix Support**: Implement sparse storage and solvers (e.g., using `sprs` or `faer-sparse`) for large-scale problems.
- [ ] **Matrix-Free Solvers**: Add support for iterative solvers (GMRES, CG) to enable solving large systems without explicit Jacobians.

### Phase 2: Advanced Output & Sensitivity Analysis
- [ ] **Numerical Quadrature**:
    - Add a `quadrature` method to `ODEProblem` to track integrals of form $\int g(t, y) dt$.
    - Support separate tolerances for quadrature components.
- [ ] **Forward Sensitivity Analysis**:
    - Implement a `SensitivityProblem` that automatically augments the system with variational equations.
    - Leverage Automatic Differentiation (AD) for Jacobian and sensitivity terms.
- [ ] **Adjoint Sensitivity Analysis**:
    - Implement the `AdjointProblem` as outlined in `docs/adjoint_design.md`.
    - Support for both Continuous and Discrete adjoints.
    - Implement checkpointing for memory-efficient backward passes.

### Phase 3: API Ergonomics & Specialized Solvers
- [ ] **Separate Tolerances**: Refactor `Tolerance` to allow per-component or per-group (state, quadrature, sensitivity) tolerances.
- [ ] **Implicit Solvers**: Expand the suite of BDF and SDIRK solvers for stiff equations and DAEs.
- [ ] **Stochastic Delay Equations (SDDEs)**: Combine DDE and SDE capabilities for more complex modeling.

## Technical Goals
- **Maintain High Performance**: Ensure that sensitivity analysis and quadrature do not introduce unnecessary overhead when not in use.
- **AD-First Design**: Deepen integration with dual numbers and other AD tools to minimize the need for manual Jacobian implementation.
- **Zero-Cost Abstractions**: Use Rust's trait system to provide a flexible API that compiles down to efficient code.
