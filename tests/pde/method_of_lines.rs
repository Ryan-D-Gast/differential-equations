use differential_equations::prelude::*;

struct HeatEquation {
    alpha: f64,
}

impl PDE for HeatEquation {
    fn flux(&self, _t: f64, _x: &[f64; 1], _u: &f64, grad_u: &[f64; 1], flux: &mut [f64; 1]) {
        flux[0] = self.alpha * grad_u[0];
    }
}

struct CoupledDiffusion {
    diffusivity: [f64; 2],
}

impl PDE<f64, Vec<f64>, 1> for CoupledDiffusion {
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

#[test]
fn method_of_lines_matches_heat_equation_mode() {
    let heat = HeatEquation { alpha: 0.1 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [41]);
    let boundary = BoundaryConditions::<f64, f64, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), 0.0)
        .dirichlet(BoundaryFace::upper(0), 0.0)
        .build()
        .expect("all boundary faces should be specified");
    let u0: Vec<f64> = grid
        .points()
        .map(|point| (std::f64::consts::PI * point[0]).sin())
        .collect();

    let solution = IVP::pde(&heat, 0.0, 0.02, u0)
        .space(MethodOfLines::finite_difference(grid.clone()).boundary(boundary))
        .method(ExplicitRungeKutta::rk4(1.0e-4))
        .solve()
        .expect("heat equation should solve");

    let uf = solution
        .y
        .last()
        .expect("solution should include final state");
    let decay = (-heat.alpha * std::f64::consts::PI.powi(2) * 0.02).exp();

    for (i, x) in grid.points().map(|point| point[0]).enumerate() {
        let expected = decay * (std::f64::consts::PI * x).sin();
        assert!(
            (uf[i] - expected).abs() < 5.0e-4,
            "node {i}: expected {expected}, got {}",
            uf[i]
        );
    }
}

#[test]
fn dirichlet_boundaries_are_held_fixed() {
    let heat = HeatEquation { alpha: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let boundary = BoundaryConditions::<f64, f64, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), 2.0)
        .dirichlet(BoundaryFace::upper(0), -1.0)
        .build()
        .expect("all boundary faces should be specified");
    let system = MethodOfLines::finite_difference(grid)
        .boundary(boundary)
        .discretize(&heat);

    let y = vec![2.0, 1.0, 0.0, -0.5, -1.0];
    let mut dydt = vec![0.0; y.len()];
    system.diff(0.0, &y, &mut dydt);

    assert_eq!(dydt[0], 0.0);
    assert_eq!(dydt[y.len() - 1], 0.0);
}

#[test]
fn method_of_lines_accepts_fixed_size_state() {
    let heat = HeatEquation { alpha: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let boundary = BoundaryConditions::<f64, f64, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), 0.0)
        .dirichlet(BoundaryFace::upper(0), 0.0)
        .build()
        .expect("all boundary faces should be specified");
    let system = MethodOfLines::finite_difference(grid)
        .boundary(boundary)
        .discretize::<_, [f64; 5]>(&heat);

    let y = [0.0, 1.0, 0.0, -1.0, 0.0];
    let mut dydt = [0.0; 5];
    system.diff(0.0, &y, &mut dydt);

    assert_eq!(dydt[0], 0.0);
    assert_eq!(dydt[4], 0.0);
    assert!(dydt[2].abs() < 1.0e-12);
}

#[test]
fn method_of_lines_accepts_vector_field_state() {
    let system = CoupledDiffusion {
        diffusivity: [1.0, 2.0],
    };
    let grid = StructuredGrid::uniform([0.0], [1.0], [3]);
    let boundary = BoundaryConditions::<f64, Vec<f64>, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), vec![0.0, 1.0])
        .dirichlet(BoundaryFace::upper(0), vec![1.0, 0.0])
        .build()
        .expect("all boundary faces should be specified");
    let semi_discrete = MethodOfLines::finite_difference_with_field(grid, vec![0.0; 2])
        .boundary(boundary)
        .discretize::<_, Vec<f64>>(&system);

    let y = vec![0.0, 1.0, 0.5, 0.25, 1.0, 0.0];
    let mut dydt = vec![0.0; y.len()];
    semi_discrete.diff(0.0, &y, &mut dydt);

    assert_eq!(dydt[0], 0.0);
    assert_eq!(dydt[1], 0.0);
    assert_eq!(dydt[4], 0.0);
    assert_eq!(dydt[5], 0.0);
    assert!((dydt[2] + 0.25).abs() < 1.0e-12);
    assert!((dydt[3] - 4.25).abs() < 1.0e-12);
}

#[test]
fn neumann_boundary_uses_prescribed_gradient() {
    let heat = HeatEquation { alpha: 1.0 };
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let dx = grid.dx(0);
    let boundary = BoundaryConditions::<f64, f64, 1>::neumann_all(0.0);
    let system = MethodOfLines::finite_difference(grid)
        .boundary(boundary)
        .discretize(&heat);

    let y = vec![1.0, 2.0, 2.0, 2.0, 2.0];
    let mut dydt = vec![0.0; y.len()];
    system.diff(0.0, &y, &mut dydt);

    assert!((dydt[0] - (1.0 / dx) / dx).abs() < 1.0e-12);
}

#[test]
fn boundary_builder_rejects_missing_faces() {
    let error = BoundaryConditions::<f64, f64, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), 0.0)
        .build()
        .expect_err("upper boundary should be required");

    assert_eq!(error.face, BoundaryFace::upper(0));
}

#[test]
fn boundary_builder_error_implements_error_trait() {
    let error = BoundaryConditions::<f64, f64, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), 0.0)
        .build()
        .expect_err("upper boundary should be required");

    let formatted = format!("{}", error);
    assert!(formatted.contains("Boundary conditions are incomplete"));

    // Verify it can be converted to Box<dyn std::error::Error>
    let dyn_error: Box<dyn std::error::Error> = Box::new(error);
    assert!(dyn_error.to_string().contains("missing face"));
}
