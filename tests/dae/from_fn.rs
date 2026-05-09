use differential_equations::prelude::*;

#[test]
fn test_dae_from_fn() {
    let t0: f64 = 0.0;
    let tf: f64 = 1.0;
    let y0: [f64; 2] = [1.0, 1.0];

    let solution = IVP::dae_from_fn(
        |_t, y, f| {
            f[0] = -y[0];
            f[1] = y[0] - y[1];
        },
        |m| {
            m[(0, 0)] = 1.0;
            m[(1, 1)] = 0.0; // y[1] is algebraic
        },
        t0,
        tf,
        y0,
    )
    .method(ImplicitRungeKutta::radau5())
    .solve()
    .unwrap();

    let final_y = solution.y.last().unwrap();
    assert!((final_y[0] - 0.3678_f64).abs() < 1e-1);
}
