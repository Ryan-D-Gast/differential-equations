use std::f64::consts::PI;

use differential_equations::prelude::*;

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

#[test]
fn yee_grid_standing_wave_pec() {
    // 2D TM standing wave in a PEC cavity
    // domain: [0, 1] x [0, 1]
    // Wave speed c = 1
    // Exact solution for TM_11 mode:
    // E_z(x, y, t) = sin(PI * x) * sin(PI * y) * cos(omega * t)
    // where omega = c * PI * sqrt(1^2 + 1^2) = c * PI * sqrt(2)

    let c2 = 1.0;
    let omega = PI * std::f64::consts::SQRT_2;
    let period = 2.0 * PI / omega;

    let equation = YeeCurlSystem;
    let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [21, 21]);

    let boundary = BoundaryConditions::dirichlet_all(vec![0.0; 3]);

    let mut u0 = vec![0.0; grid.len() * 3];
    for (i, [x, y]) in grid.points().enumerate() {
        u0[i * 3] = (PI * x).sin() * (PI * y).sin();
    }

    let t_final = period;

    let solution = IVP::pde(&equation, 0.0, t_final, u0.clone())
        .space(
            YeeGrid::uniform_2d(grid.clone(), vec![0.0; 3])
                .boundary(boundary)
                .wave_speed_squared(c2),
        )
        .method(ExplicitRungeKutta::rk4(0.001))
        .solve()
        .expect("Failed to solve Maxwell equations");

    let final_state = solution
        .y
        .last()
        .expect("solution should include final state");

    let mut max_error = 0.0_f64;
    for i in 0..grid.len() {
        let e_z_final = final_state[i * 3];
        let e_z_initial = u0[i * 3];
        let error = (e_z_final - e_z_initial).abs();
        if error > max_error {
            max_error = error;
        }
    }

    assert!(max_error < 0.05, "Max error too large: {}", max_error);
}
