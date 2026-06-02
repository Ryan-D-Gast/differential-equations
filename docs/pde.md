# Partial Differential Equations

PDE support starts with the method of lines. The spatial domain is discretized first, producing a finite-dimensional ODE system that is integrated by the existing IVP solvers.

The first supported form is a conservative PDE on a uniform structured grid:

```text
u_t = div(flux(t, x, u, grad_u)) + source(t, x, u)
```

Implement [`PDE`] for the physical equation, define a `StructuredGrid`, choose boundary conditions, and pass a `MethodOfLines` spatial discretization to `IVP::pde`.

```rust
use differential_equations::prelude::*;

struct HeatEquation {
    alpha: f64,
}

impl PDE for HeatEquation {
    fn flux(&self, _t: f64, _x: &[f64; 1], _u: &f64, grad_u: &[f64; 1], flux: &mut [f64; 1]) {
        flux[0] = self.alpha * grad_u[0];
    }
}

let heat = HeatEquation { alpha: 0.1 };
let grid = StructuredGrid::uniform([0.0], [1.0], [51]);
let boundary = BoundaryConditions::<f64, f64, 1>::builder()
    .dirichlet(BoundaryFace::lower(0), 0.0)
    .dirichlet(BoundaryFace::upper(0), 0.0)
    .build()?;
let u0 = grid.points().map(|point| (std::f64::consts::PI * point[0]).sin()).collect();

let solution = IVP::pde(&heat, 0.0, 0.05, u0)
    .space(MethodOfLines::finite_difference(grid).boundary(boundary))
    .method(ExplicitRungeKutta::rk4(5.0e-5))
    .solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For lower-level control, discretize first and then use the regular ODE API:

```rust
# use differential_equations::prelude::*;
# struct HeatEquation { alpha: f64 }
# impl PDE for HeatEquation {
#     fn flux(&self, _t: f64, _x: &[f64; 1], _u: &f64, grad_u: &[f64; 1], flux: &mut [f64; 1]) {
#         flux[0] = self.alpha * grad_u[0];
#     }
# }
# let heat = HeatEquation { alpha: 0.1 };
# let grid = StructuredGrid::uniform([0.0], [1.0], [51]);
# let boundary = BoundaryConditions::<f64, f64, 1>::builder()
#     .dirichlet(BoundaryFace::lower(0), 0.0)
#     .dirichlet(BoundaryFace::upper(0), 0.0)
#     .build()?;
# let u0 = grid.points().map(|point| (std::f64::consts::PI * point[0]).sin()).collect();
let semi_discrete = MethodOfLines::finite_difference(grid)
    .boundary(boundary)
    .discretize(&heat);

let solution = IVP::ode(&semi_discrete, 0.0, 0.05, u0)
    .method(ExplicitRungeKutta::rk4(5.0e-5))
    .solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Dirichlet boundaries hold endpoint values fixed. Neumann boundaries prescribe endpoint gradients and are included in the boundary flux balance.

`BoundaryConditions<T, U, D>` is complete by construction: every lower and upper
face must have exactly one condition. Use `BoundaryConditions::builder()` for
mixed faces, `BoundaryConditions::dirichlet_all(value)` for all-Dirichlet
domains, or `BoundaryConditions::neumann_all(value)` for all-Neumann domains.
This intentionally permits valid mixed boundary-value problems while rejecting
accidentally incomplete public boundary specifications.

Vector fields use the same trait with a local field type `U: State<T>`. The full
solver state is still a flat `State<T>` laid out by node and component:

```text
[u0_component0, u0_component1, u1_component0, u1_component1, ...]
```

For example, a two-component reaction-diffusion system can use `Vec<f64>` as its
local field type:

```rust
use differential_equations::prelude::*;

struct ReactionDiffusion {
    diffusivity: [f64; 2],
}

impl PDE<f64, Vec<f64>, 1> for ReactionDiffusion {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 1],
        _u: &Vec<f64>,
        grad_u: &[Vec<f64>; 1],
        flux: &mut [Vec<f64>; 1],
    ) {
        flux[0][0] = self.diffusivity[0] * grad_u[0][0];
        flux[0][1] = self.diffusivity[1] * grad_u[0][1];
    }

    fn source(&self, _t: f64, _x: &[f64; 1], u: &Vec<f64>, source: &mut Vec<f64>) {
        source[0] = u[1] - u[0];
        source[1] = u[0] - u[1];
    }
}
# let equation = ReactionDiffusion { diffusivity: [1.0, 0.5] };
# let grid = StructuredGrid::uniform([0.0], [1.0], [11]);
# let boundary = BoundaryConditions::<f64, Vec<f64>, 1>::builder()
#     .dirichlet(BoundaryFace::lower(0), vec![0.0, 1.0])
#     .dirichlet(BoundaryFace::upper(0), vec![1.0, 0.0])
#     .build()?;
# let u0 = vec![0.0; grid.len() * 2];
# let _solution = IVP::pde(&equation, 0.0, 0.1, u0)
#     .space(MethodOfLines::finite_volume_with_field(grid, vec![0.0; 2]).boundary(boundary))
#     .method(ExplicitRungeKutta::rk4(1.0e-4))
#     .solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use the PDE API when the model has at least one spatial derivative and one time derivative. Use the BVP API for one-dimensional ODE boundary-value problems, where the independent variable is a single coordinate and the solver is finding a trajectory that satisfies endpoint residuals.

See `examples/pde/` for complete runnable examples:

- `01_heat_equation`: scalar heat equation with `U = f64`
- `02_maxwell`: Maxwell wave system with `U = Vec<f64>` using generic method of lines
- `03_compressible_navier_stokes`: conserved-variable flow example with `U = Vec<f64>`
- `04_finite_volume_advection`: scalar advection using the finite-volume backend
- `04_maxwell_yee`: Maxwell wave system with `U = Vec<f64>` using the staggered Yee grid

## Spatial Backends

The `IVP::pde(...).space(...)` call accepts any backend implementing `SpatialDiscretization`.

- **MethodOfLines**: Generic finite-difference and finite-volume schemes for general advection-diffusion PDEs.
- **FiniteVolume**: Cell-centered finite-volume backend with explicit control over numerical fluxes, MUSCL reconstruction, and limiters.
- **YeeGrid**: A staggered FDTD/Yee grid specialized for Maxwell's equations. Use this when solving Maxwell's wave equations, as it naturally handles curl operators and provides superior wave propagation and stability properties compared to co-located generic schemes.

For conservation laws needing robust shock capturing, use `FiniteVolume`:

```rust
# use differential_equations::prelude::*;
# struct LinearAdvection { a: f64 }
# impl PDE for LinearAdvection {
#     fn flux(&self, _t: f64, _x: &[f64; 1], u: &f64, _grad_u: &[f64; 1], flux: &mut [f64; 1]) {
#         flux[0] = self.a * u;
#     }
# }
# let advection = LinearAdvection { a: 1.0 };
# let grid = StructuredGrid::uniform([0.0], [1.0], [10]);
# let boundary = BoundaryConditions::neumann_all(0.0);
# let u0 = vec![0.0; grid.len()];
let solution = IVP::pde(&advection, 0.0, 0.5, u0)
    .space(
        FiniteVolume::structured(grid)
            .boundary(boundary)
            .reconstruction(Reconstruction::Muscl)
            .limiter(Limiter::Minmod)
            .flux(NumericalFlux::Rusanov { max_speed: 1.0 }),
    )
    .method(ExplicitRungeKutta::ssp_rk3(0.01))
    .solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
