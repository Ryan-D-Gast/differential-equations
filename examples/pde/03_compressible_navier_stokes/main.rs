//! Example 03: Compressible Navier-Stokes equations with method of lines.
//!
//! This example uses conserved variables `[rho, rho u, rho v, E]` on a
//! structured grid. It is a smooth vector-field demonstration for the
//! backend-selectable PDE API, not a shock-capturing CFD solver.

use differential_equations::prelude::*;
use quill::prelude::*;

struct CompressibleNavierStokes {
    gamma: f64,
    viscosity: f64,
}

impl CompressibleNavierStokes {
    fn primitive(&self, state: &[f64]) -> (f64, f64, f64, f64) {
        let rho = state[0].max(1.0e-12);
        let rho_u = state[1];
        let rho_v = state[2];
        let energy = state[3].max(1.0e-12);
        let u = rho_u / rho;
        let v = rho_v / rho;
        let kinetic = 0.5 * rho * (u * u + v * v);
        let pressure = (self.gamma - 1.0) * (energy - kinetic).max(1.0e-12);
        (rho, u, v, pressure)
    }

    fn conserved_from_primitive(&self, rho: f64, u: f64, v: f64, pressure: f64) -> [f64; 4] {
        let energy = pressure / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
        [rho, rho * u, rho * v, energy]
    }
}

impl PDE<f64, Vec<f64>, 2> for CompressibleNavierStokes {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 2],
        state: &Vec<f64>,
        grad_state: &[Vec<f64>; 2],
        flux: &mut [Vec<f64>; 2],
    ) {
        let (rho, u, v, pressure) = self.primitive(state);
        let du_dx = (grad_state[0][1] * rho - state[1] * grad_state[0][0]) / (rho * rho);
        let du_dy = (grad_state[1][1] * rho - state[1] * grad_state[1][0]) / (rho * rho);
        let dv_dx = (grad_state[0][2] * rho - state[2] * grad_state[0][0]) / (rho * rho);
        let dv_dy = (grad_state[1][2] * rho - state[2] * grad_state[1][0]) / (rho * rho);
        let tau_xx = 2.0 * self.viscosity * du_dx;
        let tau_xy = self.viscosity * (du_dy + dv_dx);
        let tau_yy = 2.0 * self.viscosity * dv_dy;
        let energy = state[3];

        // Conservative laws are normally q_t + F(q)_x + G(q)_y = viscous terms.
        // The adapter computes q_t = div(flux), so inviscid fluxes are negated.
        flux[0][0] = -state[1];
        flux[0][1] = -(state[1] * u + pressure) + tau_xx;
        flux[0][2] = -(state[2] * u) + tau_xy;
        flux[0][3] = -((energy + pressure) * u) + tau_xx * u + tau_xy * v;

        flux[1][0] = -state[2];
        flux[1][1] = -(state[1] * v) + tau_xy;
        flux[1][2] = -(state[2] * v + pressure) + tau_yy;
        flux[1][3] = -((energy + pressure) * v) + tau_xy * u + tau_yy * v;
    }
}

fn main() {
    let ns = CompressibleNavierStokes {
        gamma: 1.4,
        viscosity: 0.001,
    };
    let grid = StructuredGrid::uniform([0.0_f64, 0.0], [1.0, 1.0], [31, 31]);
    let local_field = vec![0.0; 4];

    let initial_state_at = |x: f64, y: f64| {
        let rho = 1.0 + 0.12 * (-120.0 * ((x - 0.45).powi(2) + (y - 0.5).powi(2))).exp();
        let u = 0.03 * (2.0 * std::f64::consts::PI * y).sin();
        let v = -0.03 * (2.0 * std::f64::consts::PI * x).sin();
        ns.conserved_from_primitive(rho, u, v, 1.0)
    };

    let boundary = BoundaryConditions::neumann_all(vec![0.0; 4]);

    let mut u0 = Vec::with_capacity(grid.len() * local_field.len());
    for [x, y] in grid.points() {
        u0.extend(initial_state_at(x, y));
    }

    let solution = IVP::pde(&ns, 0.0, 0.01, u0)
        .space(
            MethodOfLines::finite_volume_with_field(grid.clone(), local_field).boundary(boundary),
        )
        .method(ExplicitRungeKutta::rk4(5.0e-5))
        .even(0.002)
        .solve()
        .expect("compressible Navier-Stokes-style system should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");
    let [nx, ny] = grid.nodes();
    let j_mid = ny / 2;

    Plot::builder()
        .title("Compressible Navier-Stokes-Style System")
        .x_label("Position x at y = 0.5")
        .y_label("Primitive variable")
        .legend(Legend::TopRightInside)
        .data([
            Series::builder()
                .name("Density rho")
                .color("Blue")
                .data(
                    (0..nx)
                        .map(|i| {
                            let node = grid.flat_index([i, j_mid]);
                            (grid.point(node)[0], final_state[4 * node])
                        })
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::None)
                .line(Line::Solid)
                .build(),
            Series::builder()
                .name("Velocity u")
                .color("Red")
                .data(
                    (0..nx)
                        .map(|i| {
                            let node = grid.flat_index([i, j_mid]);
                            let offset = 4 * node;
                            (
                                grid.point(node)[0],
                                final_state[offset + 1] / final_state[offset],
                            )
                        })
                        .collect::<Vec<_>>(),
                )
                .marker(Marker::None)
                .line(Line::Dashed)
                .build(),
        ])
        .build()
        .to_svg("examples/pde/03_compressible_navier_stokes/compressible_navier_stokes.svg")
        .expect("failed to save plot as SVG");

    println!("Compressible Navier-Stokes-style system solved successfully");
}
