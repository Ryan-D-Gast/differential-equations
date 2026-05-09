use differential_equations::prelude::*;

#[test]
fn test_dde_from_fn() {
    let t0: f64 = 0.0;
    let tf: f64 = 1.0;
    let y0: f64 = 1.0;

    let solution = IVP::dde_from_fn(
        |_t, _y, yd: &[f64; 1], dydt| *dydt = -yd[0], // dy/dt = -y(t - 1)
        |_t, _y, lags| lags[0] = 1.0,                 // constant lag of 1.0
        t0,
        tf,
        y0,
        |_t| 1.0, // history function y(t) = 1.0 for t <= 0
    )
    .method(ExplicitRungeKutta::euler(0.1))
    .solve()
    .unwrap();

    let final_t = solution.t.last().unwrap();
    assert!((*final_t - tf).abs() < 1e-10);
}
