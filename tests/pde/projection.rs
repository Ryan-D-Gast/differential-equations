use differential_equations::{ode::ODE, prelude::*};

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

    let [nx, ny] = grid.nodes();
    let mut max_divergence = 0.0_f64;
    for i in 1..(nx - 1) {
        for j in 1..(ny - 1) {
            let node = grid.flat_index([i, j]);
            let du_dx = (dudt[2 * grid.flat_index([i + 1, j])]
                - dudt[2 * grid.flat_index([i - 1, j])])
                / (2.0 * grid.dx(0));
            let dv_dy = (dudt[2 * grid.flat_index([i, j + 1]) + 1]
                - dudt[2 * grid.flat_index([i, j - 1]) + 1])
                / (2.0 * grid.dx(1));
            max_divergence = max_divergence.max((du_dx + dv_dy).abs());
            assert!(dudt[2 * node].is_finite());
            assert!(dudt[2 * node + 1].is_finite());
        }
    }

    assert!(max_divergence < 1.0);
}
