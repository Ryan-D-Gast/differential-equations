//! Example 04: finite-volume advection with MUSCL reconstruction and limiters.
//!
//! This example solves the linear advection equation:
//!
//! u_t + a u_x = 0
//!
//! using the finite-volume spatial backend.

use differential_equations::prelude::*;
use quill::prelude::*;

struct LinearAdvection {
    a: f64,
}

impl PDE for LinearAdvection {
    fn flux(&self, _t: f64, _x: &[f64; 1], u: &f64, _grad_u: &[f64; 1], flux: &mut [f64; 1]) {
        flux[0] = self.a * u;
    }
}

fn main() {
    let advection = LinearAdvection { a: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [100]);

    // Smooth pulse that stays away from the boundaries over this short run.
    let u0: Vec<f64> = grid
        .points()
        .map(|point| gaussian(point[0], 0.25, 0.04))
        .collect();

    let boundary = BoundaryConditions::neumann_all(0.0);

    let tf = 0.2;

    let solution = IVP::pde(&advection, 0.0, tf, u0)
        .space(
            FiniteVolume::structured(grid.clone())
                .boundary(boundary)
                .reconstruction(Reconstruction::Muscl)
                .limiter(Limiter::Minmod)
                .flux(NumericalFlux::Rusanov {
                    max_speed: advection.a.abs(),
                }),
        )
        // Use SSP-RK3 for stability with finite volume
        .method(ExplicitRungeKutta::ssp_rk3(0.005))
        .even(0.05)
        .solve()
        .expect("advection equation should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");

    Plot::builder()
        .title("Linear Advection by Finite Volume")
        .x_label("Position x")
        .y_label("u")
        .legend(Legend::TopRightInside)
        .data([
            Series::builder()
                .name("Numerical solution")
                .color("Blue")
                .data(
                    grid.points()
                        .map(|point| point[0])
                        .zip(final_state.iter())
                        .map(|(x, u)| (x, *u))
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::Circle)
                .marker_size(3.0)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Advected initial pulse")
                .color("Red")
                .data(
                    grid.points()
                        .map(|point| point[0])
                        .map(|x| {
                            let x_shifted = x - advection.a * tf;
                            (x, gaussian(x_shifted, 0.25, 0.04))
                        })
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/pde/04_finite_volume_advection/finite_volume_advection.svg")
        .expect("failed to save plot as SVG");

    println!("Finite volume advection solved successfully");
}

fn gaussian(x: f64, center: f64, width: f64) -> f64 {
    (-((x - center) / width).powi(2)).exp()
}
