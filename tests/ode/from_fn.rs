use differential_equations::prelude::*;

#[test]
fn test_ode_from_fn() {
    let t0: f64 = 0.0;
    let tf: f64 = 1.0;
    let y0: f64 = 1.0;

    let solution = IVP::ode_from_fn(|_t, y, dydt| *dydt = *y, t0, tf, y0)
        .method(ExplicitRungeKutta::euler(0.1))
        .solve()
        .unwrap();

    let final_y = solution.y.last().unwrap();
    // Exact solution is e^t, so e^1 = 2.718...
    // Euler approximation is (1 + 0.1)^10 = 2.5937...
    assert!((*final_y - 2.5937_f64).abs() < 1e-3);
}
