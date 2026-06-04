use differential_equations::prelude::*;
use nalgebra::SVector;

// A simple decay model: dy/dt = -k * y
#[derive(Clone)]
struct Decay {
    k: f64,
}

impl ODE<f64, SVector<f64, 1>> for Decay {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = -self.k * y[0];
    }
}

impl ParametrizedODE<f64, SVector<f64, 1>, SVector<f64, 1>> for Decay {
    fn parameters(&self) -> SVector<f64, 1> {
        SVector::from([self.k])
    }

    fn jacobian_p(&self, _t: f64, y: &SVector<f64, 1>, j: &mut Matrix<f64>) {
        j[(0, 0)] = -y[0];
    }
}

#[test]
fn test_forward_sensitivity_borrowed() {
    let decay = Decay { k: 1.0 };
    let y0 = SVector::<f64, 1>::from([1.0]);
    let y0_aug = SVector::<f64, 2>::from([1.0, 0.0]);
    let t0 = 0.0;
    let tf = 2.0;

    let solution = IVP::ode(&decay, t0, tf, y0)
        .forward_sensitivity(y0_aug)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let y_final_aug = solution.y.last().unwrap();
    let expected_val = (-2.0_f64).exp();
    let expected_sens = -2.0_f64 * expected_val;

    assert!((y_final_aug[0] - expected_val).abs() < 1e-6);
    assert!((y_final_aug[1] - expected_sens).abs() < 1e-6);
}

#[test]
fn test_forward_sensitivity_owned() {
    let decay = Decay { k: 1.0 };
    let y0 = SVector::<f64, 1>::from([1.0]);
    let y0_aug = SVector::<f64, 2>::from([1.0, 0.0]);
    let t0 = 0.0;
    let tf = 2.0;

    let solution = IVP::ode_owned(decay, t0, tf, y0)
        .forward_sensitivity(y0_aug)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let y_final_aug = solution.y.last().unwrap();
    let expected_val = (-2.0_f64).exp();
    let expected_sens = -2.0_f64 * expected_val;

    assert!((y_final_aug[0] - expected_val).abs() < 1e-6);
    assert!((y_final_aug[1] - expected_sens).abs() < 1e-6);
}

#[test]
fn test_forward_sensitivity_from_fn() {
    let k = 1.0;
    let y0 = SVector::<f64, 1>::from([1.0]);
    let y0_aug = SVector::<f64, 2>::from([1.0, 0.0]);
    let t0 = 0.0;
    let tf = 2.0;

    let solution = IVP::ode_from_fn(
        move |_t, y, dydt| {
            dydt[0] = -k * y[0];
        },
        t0,
        tf,
        y0,
    )
    .forward_sensitivity_from_fn(
        |_t, y, j| {
            j[(0, 0)] = -y[0];
        },
        SVector::<f64, 1>::from([k]),
        y0_aug,
    )
    .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
    .solve()
    .unwrap();

    let y_final_aug = solution.y.last().unwrap();
    let expected_val = (-2.0_f64).exp();
    let expected_sens = -2.0_f64 * expected_val;

    assert!((y_final_aug[0] - expected_val).abs() < 1e-6);
    assert!((y_final_aug[1] - expected_sens).abs() < 1e-6);
}

#[test]
fn test_adjoint_sensitivity_borrowed() {
    let decay = Decay { k: 1.0 };
    let y0 = SVector::<f64, 1>::from([1.0]);
    let t0 = 0.0;
    let tf = 2.0;

    let forward_solution = IVP::ode(&decay, t0, tf, y0)
        .dense(10)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    // Cost function: G = 0.5 * (y(tf) - 0.5)^2
    let y_final = forward_solution.y.last().unwrap();
    let dg_dy = y_final[0] - 0.5;
    let adjoint_y0 = SVector::<f64, 2>::from([dg_dy, 0.0]);

    let adjoint_solution = forward_solution
        .adjoint_sensitivity(&decay, adjoint_y0)
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let final_adjoint_state = adjoint_solution.y.last().unwrap();
    let numerical_gradient = final_adjoint_state[1];

    // Analytical solution:
    // y(tf) = e^(-k*tf)
    // dy(tf)/dk = -tf * e^(-k*tf)
    // dG/dk = (y(tf) - 0.5) * dy(tf)/dk
    let expected_val = (-2.0_f64).exp();
    let dy_dk = -2.0 * expected_val;
    let expected_gradient = (expected_val - 0.5) * dy_dk;

    assert!((numerical_gradient - expected_gradient).abs() < 1e-6);
}

#[test]
fn test_adjoint_sensitivity_from_fn() {
    let k = 1.0;
    let y0 = SVector::<f64, 1>::from([1.0]);
    let t0 = 0.0;
    let tf = 2.0;

    let forward_solution = IVP::ode_from_fn(
        move |_t, y, dydt| {
            dydt[0] = -k * y[0];
        },
        t0,
        tf,
        y0,
    )
    .dense(10)
    .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
    .solve()
    .unwrap();

    let y_final = forward_solution.y.last().unwrap();
    let dg_dy = y_final[0] - 0.5;
    let adjoint_y0 = SVector::<f64, 2>::from([dg_dy, 0.0]);

    let adjoint_solution = forward_solution
        .adjoint_sensitivity_from_fn(
            move |_t, y, dydt| {
                dydt[0] = -k * y[0];
            },
            |_t, y, j| {
                j[(0, 0)] = -y[0];
            },
            SVector::<f64, 1>::from([k]),
            adjoint_y0,
        )
        .method(ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-8))
        .solve()
        .unwrap();

    let final_adjoint_state = adjoint_solution.y.last().unwrap();
    let numerical_gradient = final_adjoint_state[1];

    let expected_val = (-2.0_f64).exp();
    let dy_dk = -2.0 * expected_val;
    let expected_gradient = (expected_val - 0.5) * dy_dk;

    assert!((numerical_gradient - expected_gradient).abs() < 1e-6);
}
