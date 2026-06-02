//! Example 04: incompressible Navier-Stokes projection backend.
//!
//! This runs a smooth divergence-free cavity vortex through the projection
//! backend and plots the final centerline velocity.

use std::f64::consts::PI;

use differential_equations::prelude::*;
use quill::prelude::*;

struct ViscousVelocity {
    viscosity: f64,
}

impl PDE<f64, Vec<f64>, 2> for ViscousVelocity {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 2],
        _u: &Vec<f64>,
        grad_u: &[Vec<f64>; 2],
        flux: &mut [Vec<f64>; 2],
    ) {
        flux[0][0] = self.viscosity * grad_u[0][0];
        flux[0][1] = self.viscosity * grad_u[0][1];
        flux[1][0] = self.viscosity * grad_u[1][0];
        flux[1][1] = self.viscosity * grad_u[1][1];
    }
}

fn main() {
    let equation = ViscousVelocity { viscosity: 0.01 };
    let grid = StructuredGrid::uniform([0.0_f64, 0.0], [1.0, 1.0], [21, 21]);

    let mut u0 = Vec::with_capacity(2 * grid.len());
    for [x, y] in grid.points() {
        let u = (PI * x).sin().powi(2) * (2.0 * PI * y).sin();
        let v = -(PI * y).sin().powi(2) * (2.0 * PI * x).sin();
        u0.extend([u, v]);
    }
    let boundary = BoundaryConditions::dirichlet_all(vec![0.0; 2]);

    let solution = IVP::pde(&equation, 0.0, 0.05, u0)
        .space(ProjectionMethod::uniform(grid.clone()).boundary(boundary))
        .method(ExplicitRungeKutta::rk4(1.0e-3))
        .even(0.01)
        .solve()
        .expect("projection method should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");
    let [nx, ny] = grid.nodes();
    let j_mid = ny / 2;

    Plot::builder()
        .title("Incompressible Velocity by Projection Method")
        .x_label("Position x at y = 0.5")
        .y_label("v velocity")
        .legend(Legend::TopRightInside)
        .data([Series::builder()
            .name("v centerline")
            .color("Blue")
            .data(
                (0..nx)
                    .map(|i| {
                        let node = grid.flat_index([i, j_mid]);
                        (grid.point(node)[0], final_state[2 * node + 1])
                    })
                    .collect::<Vec<_>>(),
            )
            .marker(Marker::None)
            .line(Line::Solid)
            .build()])
        .build()
        .to_svg("examples/pde/06_incompressible_navier_stokes/navier_stokes.svg")
        .expect("failed to save plot as SVG");

    println!("Incompressible Navier-Stokes projection example solved successfully");
}
