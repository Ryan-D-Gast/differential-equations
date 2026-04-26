use differential_equations::prelude::*;
use nalgebra::{Vector1, vector};

struct ExponentialDecay;

impl ODE<f64, Vector1<f64>> for ExponentialDecay {
    fn diff(&self, _t: f64, y: &Vector1<f64>, dydt: &mut Vector1<f64>) {
        dydt[0] = -y[0];
    }
}

impl ODEParameters<f64, Vector1<f64>, Vector1<f64>> for ExponentialDecay {
    fn diff_with_params(
        &self,
        _t: f64,
        y: &Vector1<f64>,
        params: &Vector1<f64>,
        dydt: &mut Vector1<f64>,
    ) {
        dydt[0] = -params[0] * y[0];
    }

    fn jacobian_state(
        &self,
        _t: f64,
        _y: &Vector1<f64>,
        params: &Vector1<f64>,
        jy: &mut Matrix<f64>,
    ) {
        jy[(0, 0)] = -params[0];
    }

    fn jacobian_params(
        &self,
        _t: f64,
        y: &Vector1<f64>,
        _params: &Vector1<f64>,
        jp: &mut Matrix<f64>,
    ) {
        jp[(0, 0)] = -y[0];
    }
}

struct TerminalState;

impl AdjointCost<f64, Vector1<f64>, Vector1<f64>> for TerminalState {
    fn terminal(&self, _tf: f64, yf: &Vector1<f64>, _params: &Vector1<f64>) -> f64 {
        yf[0]
    }

    fn terminal_gradient_y(
        &self,
        _tf: f64,
        _yf: &Vector1<f64>,
        _params: &Vector1<f64>,
        grad_y: &mut Vector1<f64>,
    ) {
        grad_y[0] = 1.0;
    }
}

fn main() -> Result<(), Error<f64, Vector1<f64>>> {
    let equation = ExponentialDecay;
    let params = vector![0.5];
    let y0 = vector![1.0];
    let t0 = 0.0;
    let tf = 2.0;

    let forward_solution = Ivp::ode(&equation, t0, tf, y0)
        .method(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12))
        .forward_sensitivity(&params)
        .solve()
        .expect("forward sensitivity solve should complete");
    let fsa_final = forward_solution.y.last().expect("solution has final state");

    let cost = TerminalState;
    let forward = ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12);
    let backward = ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12);
    let adjoint = Ivp::ode(&equation, t0, tf, y0)
        .method(forward)
        .adjoint_sensitivity(&params, &cost)
        .backward_method(backward)
        .solve()?;

    println!("Forward sensitivity dy(tf)/dp: {:.8}", fsa_final[1]);
    println!("Adjoint sensitivity dJ/dp:     {:.8}", adjoint.grad_p[0]);
    println!(
        "Analytic sensitivity:          {:.8}",
        -2.0 * (-1.0_f64).exp()
    );

    Ok(())
}
