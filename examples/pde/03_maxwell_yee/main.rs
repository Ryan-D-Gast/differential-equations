//! Example 03: Maxwell equations with Yee Grid.
//!
//! This solves a transverse-magnetic 2D Maxwell system with local field
//! `[E_z, H_x, H_y]` on a structured grid using the staggered Yee scheme.
//! The plotted output is the final electric-field cross-section through the domain center.

use differential_equations::prelude::*;
use quill::prelude::*;

struct YeeCurlSystem;

impl PDE<f64, Vec<f64>, 2> for YeeCurlSystem {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 2],
        _u: &Vec<f64>,
        _grad_u: &[Vec<f64>; 2],
        _flux: &mut [Vec<f64>; 2],
    ) {
    }
}

fn main() {
    let equation = YeeCurlSystem;
    let wave_speed = 1.0;
    let grid = StructuredGrid::uniform([0.0_f64, 0.0], [1.0, 1.0], [41, 41]);
    let local_field = vec![0.0; 3];
    let boundary = BoundaryConditions::dirichlet_all(vec![0.0; 3]);

    let mut u0 = Vec::with_capacity(grid.len() * local_field.len());
    for [x, y] in grid.points() {
        let ez = (-180.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2))).exp();
        u0.extend([ez, 0.0, 0.0]);
    }

    let solution = IVP::pde(&equation, 0.0, 0.12, u0)
        .space(
            YeeGrid::uniform_2d(grid.clone(), local_field)
                .boundary(boundary)
                .wave_speed(wave_speed),
        )
        .method(ExplicitRungeKutta::rk4(2.0e-4))
        .even(0.02)
        .solve()
        .expect("Maxwell system should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");
    let [nx, ny] = grid.nodes();
    let j_mid = ny / 2;

    Plot::builder()
        .title("Maxwell Equations by Yee Grid")
        .x_label("Position x at y = 0.5")
        .y_label("Electric field E_z")
        .legend(Legend::TopRightInside)
        .data([Series::builder()
            .name("E_z cross-section")
            .color("Red")
            .data(
                (0..nx)
                    .map(|i| {
                        let node = grid.flat_index([i, j_mid]);
                        (grid.point(node)[0], final_state[3 * node])
                    })
                    .collect::<Vec<_>>(),
            )
            .marker(Marker::None)
            .line(Line::Solid)
            .build()])
        .build()
        .to_svg("examples/pde/03_maxwell_yee/maxwell_yee.svg")
        .expect("failed to save plot as SVG");

    println!("Maxwell system solved successfully with Yee Grid");
}
