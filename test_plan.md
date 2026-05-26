1. **Add `navier_stokes` module in `src/pde/mod.rs`**: Include `mod navier_stokes; pub use navier_stokes::*;`.
2. **Implement `ProjectionMethod` in `src/pde/navier_stokes/projection.rs`**: Implement the backend struct with `uniform` constructor, `boundary` modifier, and `SpatialDiscretization` trait implementation mapping it to a semi-discrete ODE. To handle pressure, we will need to perform a Poisson solve and enforce divergence-free velocity inside the `ODE::diff` method of the resulting system.
3. **Update `docs/pde.md`**: Add section about when to use the projection backend for incompressible Navier-Stokes.
4. **Create `examples/pde/04_incompressible_navier_stokes/main.rs`**: Build an example such as cavity or periodic flow.
5. **Add tests**: In `tests/pde/main.rs` or `src/pde/navier_stokes/projection.rs` add divergence reduction and boundary tests.
6. **Pre-commit**: Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
