//! Solve independent decay equations using a runtime-sized custom state.
use differential_equations::prelude::*;

#[derive(State)]
struct Samples<T> {
    values: Vec<T>,
}

struct Decay;

impl ODE<f64, Samples<f64>> for Decay {
    fn diff(&self, _t: f64, y: &Samples<f64>, dydt: &mut Samples<f64>) {
        for (out, value) in dydt.values.iter_mut().zip(&y.values) {
            *out = -*value;
        }
    }
}

fn main() {
    let solution = IVP::ode(
        &Decay,
        0.0,
        1.0,
        Samples {
            values: vec![1.0, 2.0],
        },
    )
    .method(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-10))
    .solve()
    .expect("Decay integration failed");
    let last = solution
        .y
        .last()
        .expect("Solution contains its final state");
    for (value, initial) in last.values.iter().zip([1.0, 2.0]) {
        assert!((value - initial * (-1.0_f64).exp()).abs() < 1e-8);
    }
}
