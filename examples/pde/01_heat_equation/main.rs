//! Example 01: Heat equation with method of lines.
//!
//! This example solves
//!
//! u_t = alpha u_xx
//!
//! on x in [0, 1] with homogeneous Dirichlet boundary conditions. The spatial
//! derivative is discretized with method of lines, and the resulting ODE system
//! is integrated with the existing RK4 solver.

use differential_equations::prelude::*;
use quill::prelude::*;

struct HeatEquation {
    alpha: f64,
}

impl PDE for HeatEquation {
    fn flux(&self, _t: f64, _x: &[f64; 1], _u: &f64, grad_u: &[f64; 1], flux: &mut [f64; 1]) {
        flux[0] = self.alpha * grad_u[0];
    }
}

fn main() {
    let heat = HeatEquation { alpha: 0.1 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [51]);
    let boundary = BoundaryConditions::builder()
        .dirichlet(BoundaryFace::lower(0), 0.0)
        .dirichlet(BoundaryFace::upper(0), 0.0)
        .build()
        .expect("all boundary faces should be specified");
    let u0: Vec<f64> = grid
        .points()
        .map(|point| (std::f64::consts::PI * point[0]).sin())
        .collect();
    let tf = 0.05;

    let solution = IVP::pde(&heat, 0.0, tf, u0)
        .space(MethodOfLines::finite_difference(grid.clone()).boundary(boundary))
        .method(ExplicitRungeKutta::rk4(5.0e-5))
        .even(0.01)
        .solve()
        .expect("heat equation should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");
    let decay = (-heat.alpha * std::f64::consts::PI.powi(2) * tf).exp();

    Plot::builder()
        .title("Heat Equation by Method of Lines")
        .x_label("Position x")
        .y_label("Temperature u")
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
                .name("Analytical solution")
                .color("Red")
                .data(
                    grid.points()
                        .map(|point| point[0])
                        .map(|x| (x, decay * (std::f64::consts::PI * x).sin()))
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/pde/01_heat_equation/heat_equation.svg")
        .expect("failed to save plot as SVG");

    println!("Heat equation solved successfully with method of lines");
}
