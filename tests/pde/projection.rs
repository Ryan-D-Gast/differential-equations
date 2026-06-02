use differential_equations::{ode::ODE, prelude::*};
use std::f64::consts::PI;

struct GradientDrive;

impl PDE<f64, Vec<f64>, 2> for GradientDrive {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 2],
        _u: &Vec<f64>,
        _grad_u: &[Vec<f64>; 2],
        flux: &mut [Vec<f64>; 2],
    ) {
        flux[0][0] = 1.0;
        flux[0][1] = 0.0;
        flux[1][0] = 0.0;
        flux[1][1] = 1.0;
    }

    fn source(&self, _t: f64, x: &[f64; 2], _u: &Vec<f64>, source: &mut Vec<f64>) {
        source[0] = x[0];
        source[1] = x[1];
    }
}

#[test]
fn projection_reduces_velocity_divergence() {
    let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [9, 9]);
    let system = ProjectionMethod::uniform(grid.clone()).discretize(&GradientDrive);
    let y = vec![0.0; 2 * grid.len()];
    let mut dudt = vec![0.0; 2 * grid.len()];

    system.diff(0.0, &y, &mut dudt);

    let max_divergence = max_velocity_divergence(&grid, &dudt, [0, 1], 2);
    let unprojected_divergence = 2.0;

    assert!(
        max_divergence < 0.5 * unprojected_divergence,
        "projection did not sufficiently reduce divergence: {max_divergence}"
    );

    for node in 0..grid.len() {
        assert!(dudt[2 * node].is_finite());
        assert!(dudt[2 * node + 1].is_finite());
    }
}

fn max_velocity_divergence(
    grid: &StructuredGrid<f64, 2>,
    y: &[f64],
    velocity_components: [usize; 2],
    local_len: usize,
) -> f64 {
    let [nx, ny] = grid.nodes();
    let mut max_divergence = 0.0_f64;
    for i in 1..(nx - 1) {
        for j in 1..(ny - 1) {
            let du_dx = (y[local_len * grid.flat_index([i + 1, j]) + velocity_components[0]]
                - y[local_len * grid.flat_index([i - 1, j]) + velocity_components[0]])
                / (2.0 * grid.dx(0));
            let dv_dy = (y[local_len * grid.flat_index([i, j + 1]) + velocity_components[1]]
                - y[local_len * grid.flat_index([i, j - 1]) + velocity_components[1]])
                / (2.0 * grid.dx(1));
            max_divergence = max_divergence.max((du_dx + dv_dy).abs());
        }
    }
    max_divergence
}

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

#[test]
fn projection_cavity_vortex_example_stays_finite() {
    let equation = ViscousVelocity { viscosity: 0.01 };
    let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [21, 21]);

    let mut u0 = Vec::with_capacity(2 * grid.len());
    for [x, y] in grid.points() {
        let u = (PI * x).sin().powi(2) * (2.0 * PI * y).sin();
        let v = -(PI * y).sin().powi(2) * (2.0 * PI * x).sin();
        u0.extend([u, v]);
    }
    let boundary = BoundaryConditions::dirichlet_all(vec![0.0; 2]);

    let solution = IVP::pde(&equation, 0.0, 0.05, u0)
        .space(ProjectionMethod::uniform(grid).boundary(boundary))
        .method(ExplicitRungeKutta::rk4(1.0e-3))
        .even(0.01)
        .solve()
        .expect("projection method should solve");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");
    let max_norm = final_state
        .iter()
        .fold(0.0_f64, |max_value, value| max_value.max(value.abs()));

    assert!(
        max_norm.is_finite(),
        "projection produced a non-finite state"
    );
    assert!(
        max_norm < 2.0,
        "projection example velocity grew unexpectedly: {max_norm}"
    );
}

struct TracerSource;

impl PDE<f64, Vec<f64>, 2> for TracerSource {
    fn flux(
        &self,
        _t: f64,
        _x: &[f64; 2],
        _u: &Vec<f64>,
        _grad_u: &[Vec<f64>; 2],
        _flux: &mut [Vec<f64>; 2],
    ) {
    }

    fn source(&self, _t: f64, _x: &[f64; 2], _u: &Vec<f64>, source: &mut Vec<f64>) {
        source[0] = 5.0;
    }
}

#[test]
fn projection_preserves_unselected_components() {
    let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [5, 5]);
    let system = ProjectionMethod::with_field(grid.clone(), vec![0.0; 3])
        .velocity_components([1, 2])
        .discretize(&TracerSource);
    let y = vec![0.0; 3 * grid.len()];
    let mut dudt = vec![0.0; 3 * grid.len()];

    system.diff(0.0, &y, &mut dudt);

    for node in 0..grid.len() {
        assert_eq!(dudt[3 * node], 5.0);
        assert_eq!(dudt[3 * node + 1], 0.0);
        assert_eq!(dudt[3 * node + 2], 0.0);
    }
}
