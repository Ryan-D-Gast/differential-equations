use differential_equations::{
    bvp::BVP,
    methods::{ExplicitRungeKutta, bvp::Shooting},
};

use super::systems::{HarmonicOscillatorBvp, OdeBoundary, PipeHeatTransfer};

fn assert_harmonic_solution(solution: &differential_equations::solution::Solution<f64, [f64; 2]>) {
    let (_, y_initial) = solution
        .iter()
        .next()
        .expect("solution should contain initial point");
    let (_, y_final) = solution
        .last()
        .expect("solution should contain final point");

    assert!(y_initial[0].abs() < 1e-5);
    assert!((y_initial[1] - 1.0).abs() < 1e-5);
    assert!((y_final[0] - 1.0).abs() < 1e-5);
    assert!(y_final[1].abs() < 1e-5);
}

#[test]
fn single_shooting_solves_harmonic_oscillator() {
    let problem = HarmonicOscillatorBvp { target: 1.0 };
    let method = Shooting::single(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12));

    let solution = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
        .method(method)
        .solve()
        .expect("single shooting should converge");

    assert_harmonic_solution(&solution);
}

#[test]
fn multiple_shooting_solves_harmonic_oscillator() {
    let problem = HarmonicOscillatorBvp { target: 1.0 };
    let method = Shooting::multiple(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12))
        .segments(4)
        .tolerance(1e-8);

    let solution = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
        .method(method)
        .solve()
        .expect("multiple shooting should converge");

    assert_harmonic_solution(&solution);
    assert!(solution.evals.jacobian > 0);
    assert!(solution.evals.newton > 0);
}

#[test]
fn multiple_shooting_supports_even_output() {
    let problem = HarmonicOscillatorBvp { target: 1.0 };
    let method = Shooting::multiple(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12))
        .segments(3)
        .tolerance(1e-8);

    let solution = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
        .even(std::f64::consts::FRAC_PI_2 / 4.0)
        .method(method)
        .solve()
        .expect("multiple shooting should converge with even output");

    assert_eq!(solution.t.len(), 5);
    assert!((solution.t[0] - 0.0).abs() < 1e-12);
    assert!((solution.t[4] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert_harmonic_solution(&solution);
}

#[test]
fn bvp_trait_object_can_be_solved() {
    let problem = HarmonicOscillatorBvp { target: 1.0 };
    let problem: &dyn OdeBoundary = &problem;
    let method = Shooting::single(ExplicitRungeKutta::dop853());

    let solution = BVP::ode(problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
        .method(method)
        .solve()
        .expect("trait-object BVP should converge");

    assert_harmonic_solution(&solution);
}

#[test]
fn shooting_methods_solve_pipe_heat_transfer_bvp() {
    let pipe = PipeHeatTransfer {
        ambient_temperature: 293.15,
        heat_loss_rate: 0.2,
        inlet_temperature: 373.15,
    };
    let length = 10.0;
    let initial_guess = [pipe.inlet_temperature, -5.0];

    for solution in [
        BVP::ode(&pipe, 0.0, length, initial_guess)
            .method(Shooting::single(
                ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12),
            ))
            .solve()
            .expect("single shooting should solve pipe BVP"),
        BVP::ode(&pipe, 0.0, length, initial_guess)
            .method(
                Shooting::multiple(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12))
                    .segments(5)
                    .tolerance(1e-8),
            )
            .solve()
            .expect("multiple shooting should solve pipe BVP"),
    ] {
        let (_, y_initial) = solution
            .iter()
            .next()
            .expect("solution should include initial point");
        let (_, y_final) = solution
            .last()
            .expect("solution should include final point");

        assert!((y_initial[0] - pipe.inlet_temperature).abs() < 1e-8);
        assert!(y_final[1].abs() < 1e-5);
        assert!((y_initial[1] - pipe.analytical_initial_gradient(length)).abs() < 1e-4);
        assert!((y_final[0] - pipe.analytical_outlet_temperature(length)).abs() < 1e-4);
    }
}
