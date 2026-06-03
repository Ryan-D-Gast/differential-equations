use differential_equations::{ode::ODE, prelude::*};

struct LinearAdvection {
    speed: f64,
}

impl PDE for LinearAdvection {
    fn flux(&self, _t: f64, _x: &[f64; 1], u: &f64, _grad_u: &[f64; 1], flux: &mut [f64; 1]) {
        flux[0] = self.speed * *u;
    }
}

#[test]
fn finite_volume_rusanov_matches_upwind_advection_derivative() {
    let equation = LinearAdvection { speed: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let dx = grid.dx(0);
    let system = FiniteVolume::structured(grid)
        .boundary(BoundaryConditions::neumann_all(0.0))
        .reconstruction(Reconstruction::Constant)
        .flux(NumericalFlux::Rusanov { max_speed: 1.0 })
        .discretize(&equation);

    let y = vec![1.0, 2.0, 4.0, 7.0, 11.0];
    let mut dydt = vec![0.0; y.len()];

    system.diff(0.0, &y, &mut dydt);

    let expected = [0.0, -1.0 / dx, -2.0 / dx, -3.0 / dx, -4.0 / dx];
    for (index, (actual, expected)) in dydt.iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).abs() < 1.0e-12,
            "node {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn finite_volume_total_rate_matches_boundary_flux_balance() {
    let equation = LinearAdvection { speed: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let dx = grid.dx(0);
    let system = FiniteVolume::structured(grid)
        .boundary(BoundaryConditions::neumann_all(0.0))
        .reconstruction(Reconstruction::Constant)
        .flux(NumericalFlux::Rusanov { max_speed: 1.0 })
        .discretize(&equation);

    let y = vec![1.0, 2.0, 4.0, 7.0, 11.0];
    let mut dydt = vec![0.0; y.len()];

    system.diff(0.0, &y, &mut dydt);

    let total_rate = dydt.iter().sum::<f64>() * dx;
    let expected_boundary_balance = equation.speed * y[0] - equation.speed * y[y.len() - 1];

    assert!((total_rate - expected_boundary_balance).abs() < 1.0e-12);
}
