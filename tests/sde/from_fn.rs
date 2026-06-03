use differential_equations::prelude::*;

#[test]
fn test_sde_from_fn() {
    let t0: f64 = 0.0;
    let tf: f64 = 0.1;
    let y0: f64 = 1.0;

    let ivp = IVP::sde_from_fn(
        |_t, y, dydt| *dydt = 0.5 * *y, // drift
        |_t, y, dydw| *dydw = 0.1 * *y, // diffusion
        |_dt, dw| *dw = 0.1,            // noise (deterministic for test)
        t0,
        tf,
        y0,
    );

    let solution = ivp.method(ExplicitRungeKutta::euler(0.01)).solve().unwrap();

    // We just check it runs without crashing and reaches final time
    let final_t = solution.t.last().unwrap();
    assert!((*final_t - tf).abs() < 1e-10);
}
