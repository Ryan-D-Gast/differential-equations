use differential_equations::prelude::*;
use rayon::prelude::*;

struct TestOde;
impl ODE for TestOde {
    fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = -y;
    }
}

struct TestDae;
impl DAE for TestDae {
    fn diff(&self, _t: f64, y: &f64, f: &mut f64) {
        *f = -y;
    }
    fn mass(&self, m: &mut differential_equations::linalg::Matrix<f64>) {
        m[(0, 0)] = 1.0;
    }
}

struct TestSde {
    rng: rand::rngs::StdRng,
}
use rand::SeedableRng;
impl TestSde {
    fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}
impl SDE for TestSde {
    fn drift(&self, _t: f64, y: &f64, dydt: &mut f64) {
        *dydt = -y;
    }
    fn diffusion(&self, _t: f64, y: &f64, dydw: &mut f64) {
        *dydw = y * 0.1;
    }
    fn noise(&mut self, dt: f64, dw: &mut f64) {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        *dw = normal.sample(&mut self.rng);
    }
}

struct TestDde;
impl DDE<1> for TestDde {
    fn diff(&self, _t: f64, y: &f64, _dy: &[f64; 1], dydt: &mut f64) {
        *dydt = -y;
    }
    fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
        lags[0] = 0.1;
    }
}

#[test]
fn test_parallel_solvers() {
    let t0 = 0.0;
    let tf = 1.0;
    let y0 = 1.0;

    let ode = TestOde;
    let dae = TestDae;
    let mut sdes: Vec<_> = (0..10).map(TestSde::new).collect();
    let dde = TestDde;

    // ODE
    let mut ode_ivps = Vec::new();
    for _ in 0..10 {
        ode_ivps.push(IVP::ode(&ode, t0, tf, y0).method(ExplicitRungeKutta::dop853()));
    }
    let ode_results: Vec<_> = ode_ivps.into_par_iter().map(|ivp| ivp.solve()).collect();
    assert_eq!(ode_results.len(), 10);
    assert!(ode_results.into_iter().all(|r| r.is_ok()));

    // DAE
    let mut dae_ivps = Vec::new();
    for _ in 0..10 {
        dae_ivps.push(IVP::dae(&dae, t0, tf, y0).method(ImplicitRungeKutta::radau5()));
    }
    let dae_results: Vec<_> = dae_ivps.into_par_iter().map(|ivp| ivp.solve()).collect();
    assert_eq!(dae_results.len(), 10);
    assert!(dae_results.into_iter().all(|r| r.is_ok()));

    // SDE
    let sde_ivps: Vec<_> = sdes
        .iter_mut()
        .map(|sde| IVP::sde(sde, t0, tf, y0).method(ExplicitRungeKutta::euler(0.01)))
        .collect();
    let sde_results: Vec<_> = sde_ivps.into_par_iter().map(|ivp| ivp.solve()).collect();
    assert_eq!(sde_results.len(), 10);
    assert!(sde_results.into_iter().all(|r| r.is_ok()));

    // DDE
    let mut dde_ivps = Vec::new();
    let history = |_: f64| 1.0;
    for _ in 0..10 {
        dde_ivps.push(IVP::dde(&dde, t0, tf, y0, history).method(ExplicitRungeKutta::dop853()));
    }
    let dde_results: Vec<_> = dde_ivps.into_par_iter().map(|ivp| ivp.solve()).collect();
    assert_eq!(dde_results.len(), 10);
    assert!(dde_results.into_iter().all(|r| r.is_ok()));
}
