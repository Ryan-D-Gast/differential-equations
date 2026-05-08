use differential_equations::{ivp::IVP, methods::ExplicitRungeKutta, ode::ODE};

struct ExponentialGrowth;

impl ODE<f64, Vec<f64>> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &Vec<f64>, dydt: &mut Vec<f64>) {
        dydt[0] = y[0];
    }
}

#[test]
fn fixed_rk4_solves_vec_state() {
    let solution = IVP::ode(&ExponentialGrowth, 0.0, 1.0, vec![1.0])
        .method(ExplicitRungeKutta::rk4(0.01))
        .solve()
        .unwrap();

    let yf = solution.y.last().unwrap();
    assert!((yf[0] - std::f64::consts::E).abs() < 1.0e-8);
}
