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
let boundary = BoundaryConditions::new()
    .dirichlet(BoundaryFace::lower(0), 0.0)
    .dirichlet(BoundaryFace::upper(0), 0.0);
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
# let boundary = BoundaryConditions::new()
#     .dirichlet(BoundaryFace::lower(0), 0.0)
#     .dirichlet(BoundaryFace::upper(0), 0.0);
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
# let boundary = BoundaryConditions::new()
#     .dirichlet(BoundaryFace::lower(0), vec![0.0, 1.0])
#     .dirichlet(BoundaryFace::upper(0), vec![1.0, 0.0]);
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
- `02_maxwell`: Maxwell wave system with `U = Vec<f64>`
- `03_compressible_navier_stokes`: conserved-variable flow example with `U = Vec<f64>`
- `04_incompressible_navier_stokes`: projection backend for incompressible velocity fields

## Incompressible Navier-Stokes

Use `ProjectionMethod` for two-dimensional incompressible velocity states laid
out as `[u0, v0, u1, v1, ...]`. The backend evaluates the supplied PDE terms,
solves a dense Poisson problem with the crate's LU solver, and subtracts the
pressure-potential gradient from the velocity tendency.

```rust
# use differential_equations::prelude::*;
# struct NavierStokes { viscosity: f64 }
# impl PDE<f64, Vec<f64>, 2> for NavierStokes {
#     fn flux(&self, _t: f64, _x: &[f64; 2], _u: &Vec<f64>, grad_u: &[Vec<f64>; 2], flux: &mut [Vec<f64>; 2]) {
#         flux[0][0] = self.viscosity * grad_u[0][0];
#         flux[0][1] = self.viscosity * grad_u[0][1];
#         flux[1][0] = self.viscosity * grad_u[1][0];
#         flux[1][1] = self.viscosity * grad_u[1][1];
#     }
# }
# let navier_stokes = NavierStokes { viscosity: 0.01 };
# let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [9, 9]);
# let velocity = vec![0.0; 2 * grid.len()];
let solution = IVP::pde(&navier_stokes, 0.0, 0.1, velocity)
    .space(ProjectionMethod::uniform(grid))
    .method(ExplicitRungeKutta::rk4(1.0e-3))
    .solve()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`MethodOfLines` currently offers finite-difference and finite-volume spatial
schemes. The `IVP::pde(...).space(...)` call accepts any backend implementing
`SpatialDiscretization`, so later specialized backends can plug into the same
builder API.
