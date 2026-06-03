use differential_equations::prelude::*;

#[test]
fn test_pde_from_fn_flux() {
    let alpha = 0.1;
    let heat = pde_from_fn_flux::<f64, f64, 1, _>(move |_t, _x, _u, grad_u, flux| {
        flux[0] = alpha * grad_u[0];
    });

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
    let decay = (-alpha * std::f64::consts::PI.powi(2) * 0.02).exp();

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
fn test_ivp_pde_from_fn() {
    let alpha = 0.1;
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

    let solution = IVP::pde_from_fn::<f64, 1>(
        move |_t, _x, _u, grad_u, flux| {
            flux[0] = alpha * grad_u[0];
        },
        0.0,
        0.02,
        u0,
    )
    .space(MethodOfLines::finite_difference(grid.clone()).boundary(boundary))
    .method(ExplicitRungeKutta::rk4(1.0e-4))
    .solve()
    .expect("heat equation should solve");

    let uf = solution
        .y
        .last()
        .expect("solution should include final state");
    let decay = (-alpha * std::f64::consts::PI.powi(2) * 0.02).exp();

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
fn test_pde_from_fn_coupled() {
    let diffusivity = [0.1, 0.2];
    let pde = pde_from_fn::<f64, Vec<f64>, 1, _, _>(
        move |_t, _x, _u, grad_u, flux| {
            flux[0][0] = diffusivity[0] * grad_u[0][0];
            flux[0][1] = diffusivity[1] * grad_u[0][1];
        },
        |_t, _x, u, source| {
            source[0] = u[1] - u[0];
            source[1] = u[0] - u[1];
        },
    );

    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let boundary = BoundaryConditions::<f64, Vec<f64>, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), vec![0.0; 2])
        .dirichlet(BoundaryFace::upper(0), vec![0.0; 2])
        .build()
        .expect("all boundary faces should be specified");

    let system = MethodOfLines::finite_difference_with_field(grid, vec![0.0; 2])
        .boundary(boundary)
        .discretize(pde);

    let y = vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut dydt = vec![0.0; y.len()];
    system.diff(0.0, &y, &mut dydt);

    // boundary points (first and last nodes) should stay fixed under Dirichlet
    assert_eq!(dydt[0], 0.0);
    assert_eq!(dydt[1], 0.0);
    assert_eq!(dydt[8], 0.0);
    assert_eq!(dydt[9], 0.0);
}

#[test]
fn test_ivp_pde_from_fn_with_source() {
    let diffusivity = [0.1, 0.2];
    let grid = StructuredGrid::uniform([0.0], [1.0], [5]);
    let boundary = BoundaryConditions::<f64, Vec<f64>, 1>::builder()
        .dirichlet(BoundaryFace::lower(0), vec![0.0; 2])
        .dirichlet(BoundaryFace::upper(0), vec![0.0; 2])
        .build()
        .expect("all boundary faces should be specified");

    let u0 = vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let solution = IVP::pde_from_fn_with_source::<Vec<f64>, 1>(
        move |_t, _x, _u, grad_u, flux| {
            flux[0][0] = diffusivity[0] * grad_u[0][0];
            flux[0][1] = diffusivity[1] * grad_u[0][1];
        },
        |_t, _x, u, source| {
            source[0] = u[1] - u[0];
            source[1] = u[0] - u[1];
        },
        0.0,
        0.02,
        u0,
    )
    .space(MethodOfLines::finite_difference_with_field(grid, vec![0.0; 2]).boundary(boundary))
    .method(ExplicitRungeKutta::rk4(1.0e-4))
    .solve()
    .expect("coupled diffusion should solve");

    let uf = solution
        .y
        .last()
        .expect("solution should include final state");

    assert_eq!(uf[0], 0.0);
    assert_eq!(uf[1], 0.0);
    assert_eq!(uf[8], 0.0);
    assert_eq!(uf[9], 0.0);
}
