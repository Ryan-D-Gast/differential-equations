use differential_equations::prelude::*;
use nalgebra::{vector, SVector};

/// Exponential growth with parameter k
/// dy/dt = k * y
struct ExponentialGrowth {
    k: f64,
}

impl ODE<f64, SVector<f64, 1>> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0];
    }

    fn jacobian_p(&self, _t: f64, y: &SVector<f64, 1>, jp: &mut Matrix<f64>) {
        // dy/dk = y
        jp[(0, 0)] = y[0];
    }
}

fn main() {
    let system = ExponentialGrowth { k: 2.0 };

    // Initial conditions
    let t0 = 0.0;
    let tf = 1.0;

    // For FSA, the augmented state is [y, S]
    // where S is the sensitivity matrix dy/dk.
    // For a 1D state and 1 parameter, S is just a scalar.
    // We start with S(0) = 0 because the initial state y(0) does not depend on k.
    let y0_aug = vector![1.0, 0.0];

    let y0_base = vector![1.0];
    // Since there is 1 parameter (k), we specify PRM = 1
    let fsa_problem = ForwardSensitivityProblem::<_, _, _, 1>::new(&system, y0_base);

    let method = ExplicitRungeKutta::dop853()
        .rtol(1e-10)
        .atol(1e-10);

    let problem = Ivp::ode(&fsa_problem, t0, tf, y0_aug)
        .method(method);

    let solution = match problem.solve() {
        Ok(sol) => sol,
        Err(e) => panic!("Error solving FSA problem: {:?}", e),
    };

    println!("Time \t y \t\t S (dy/dk)");
    for (t, y_aug) in solution.iter() {
        println!("{:.4} \t {:.6} \t {:.6}", t, y_aug[0], y_aug[1]);
    }

    let y_final = solution.y.last().unwrap();
    println!("\nFinal time t = {:.2}", tf);
    println!("y = {:.6} (Exact: {:.6})", y_final[0], (2.0f64 * 1.0).exp());
    println!("S = {:.6} (Exact: {:.6})", y_final[1], 1.0 * (2.0f64 * 1.0).exp());

    println!("Function evaluations: {}", solution.evals.function);
    println!("Jacobian evaluations: {}", solution.evals.jacobian);
}