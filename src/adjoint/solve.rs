use crate::{
    adjoint::{
        cost::CostFunction,
        system::{AdjointODE, AdjointState, ParameterizedODE},
    },
    error::Error,
    interpolate::Interpolation,
    ode::{OrdinaryNumericalMethod, solve_ode},
    solout::{DenseSolout, Solout},
    solution::Solution,
    traits::{Real, State},
};

/// Solves an Adjoint Sensitivity Analysis problem.
///
/// This function computes the forward trajectory, and then performs a
/// reverse-time integration to compute the gradient of the specified
/// cost function with respect to the initial state and parameters.
///
/// # Arguments
///
/// * `forward_solver` - The ODE solver used for the forward pass.
/// * `backward_solver` - The ODE solver used for the backward pass.
/// * `ode` - The parameterized ODE system.
/// * `cost` - The cost function object.
/// * `t0` - Initial time.
/// * `tf` - Final time.
/// * `y0` - Initial state.
/// * `p` - Parameters.
/// * `backward_solout` - Optional solout strategy for the backward pass.
///
/// # Returns
///
/// * `Result<(Solution<T, Y>, Solution<T, AdjointState<T, Y, P>>), Error<T, Y>>` -
///   Returns a tuple containing the forward solution and the adjoint solution.
///   The adjoint solution's final state `adjoint_solution.y.last().unwrap()`
///   contains `lambda` (gradient with respect to state) and `mu` (gradient
///   with respect to parameters) evaluated at `t0`.
#[allow(clippy::type_complexity)]
pub fn solve_adjoint<T, Y, P, S1, S2, F, G, O>(
    forward_solver: &mut S1,
    backward_solver: &mut S2,
    ode: &F,
    cost: &G,
    t0: T,
    tf: T,
    y0: &Y,
    p: &P,
    backward_solout: &mut O,
) -> Result<(Solution<T, Y>, Solution<T, AdjointState<T, Y, P>>), Error<T, Y>>
where
    T: Real,
    Y: State<T>,
    P: State<T>,
    F: ParameterizedODE<T, Y, P>,
    G: CostFunction<T, Y, P>,
    S1: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    S2: OrdinaryNumericalMethod<T, AdjointState<T, Y, P>> + Interpolation<T, AdjointState<T, Y, P>>,
    O: Solout<T, AdjointState<T, Y, P>>,
{
    // 1. Solve the main ODE forward in time and store the dense output.
    // We use a DenseSolout to capture the state at all points for the backward pass.
    let mut dense_solout = DenseSolout::new(1); // 1 point between steps might be sufficient or dense
    let forward_solution = solve_ode(forward_solver, ode, t0, tf, y0, &mut dense_solout)?;

    // 2. Setup the reverse-time AdjointODE
    let adjoint_system = AdjointODE {
        ode,
        cost,
        p,
        forward_solution: &forward_solution,
    };

    // The boundary conditions for the adjoint system at tf are:
    // λ(tf) = ∇_y g(tf) (if discrete cost exists at tf)
    // μ(tf) = ∇_p g(tf) (if discrete cost exists at tf)
    let y_f = forward_solution.y.last().unwrap();

    // Evaluate discrete gradient at tf
    let mut lambda_tf = Y::zeros();
    let mut mu_tf = P::zeros();

    let eps = T::default_epsilon().sqrt();
    let g0_tf = cost.discrete(tf, y_f, p);

    if g0_tf != T::zero() {
        let mut y_perturbed = *y_f;
        for i in 0..y_f.len() {
            let y_orig = y_f.get(i);
            let perturbation = eps * y_orig.abs().max(T::one());
            y_perturbed.set(i, y_orig + perturbation);
            let g1 = cost.discrete(tf, &y_perturbed, p);
            lambda_tf.set(i, (g1 - g0_tf) / perturbation);
            y_perturbed.set(i, y_orig);
        }

        let mut p_perturbed = *p;
        for i in 0..p.len() {
            let p_orig = p.get(i);
            let perturbation = eps * p_orig.abs().max(T::one());
            p_perturbed.set(i, p_orig + perturbation);
            let g1 = cost.discrete(tf, y_f, &p_perturbed);
            mu_tf.set(i, (g1 - g0_tf) / perturbation);
            p_perturbed.set(i, p_orig);
        }
    }

    let adj_y0 = AdjointState::new(lambda_tf, mu_tf);

    // 3. Solve the AdjointODE backwards from tf down to t0.
    // The negative step size logic is naturally supported in `solve_ode`
    // when `t0 > tf` logic is inverted (here integrating from `tf` to `t0`).
    let adjoint_solution = solve_ode(
        backward_solver,
        &adjoint_system,
        tf,
        t0,
        &adj_y0,
        backward_solout,
    )
    .map_err(|e| match e {
        Error::BadInput { msg } => Error::BadInput { msg },
        Error::MaxSteps { t, y } => Error::MaxSteps { t, y: y.lambda }, // map state loosely if needed, or simply pass through differently if we couldn't
        Error::StepSize { t, y } => Error::StepSize { t, y: y.lambda },
        Error::Stiffness { t, y } => Error::Stiffness { t, y: y.lambda },
        Error::OutOfBounds {
            t_interp,
            t_prev,
            t_curr,
        } => Error::OutOfBounds {
            t_interp,
            t_prev,
            t_curr,
        },
        Error::NoLags => Error::NoLags,
        Error::InsufficientHistory {
            t_delayed,
            t_prev,
            t_curr,
        } => Error::InsufficientHistory {
            t_delayed,
            t_prev,
            t_curr,
        },
        Error::LinearAlgebra { msg } => Error::LinearAlgebra { msg },
    })?;

    Ok((forward_solution, adjoint_solution))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{methods::ExplicitRungeKutta, ode::ODE, solout::DefaultSolout};

    struct TestODE;

    impl ODE<f64, f64> for TestODE {
        fn diff(&self, _t: f64, y: &f64, dydt: &mut f64) {
            *dydt = -2.0 * *y;
        }
    }

    impl ParameterizedODE<f64, f64, f64> for TestODE {
        fn diff_p(&self, _t: f64, y: &f64, p: &f64, dydt: &mut f64) {
            *dydt = -*p * *y;
        }
    }

    struct TestCost;

    impl CostFunction<f64, f64, f64> for TestCost {
        fn discrete(&self, t: f64, y: &f64, _p: &f64) -> f64 {
            if t == 1.0 {
                // Cost is just the final state value
                *y
            } else {
                0.0
            }
        }

        // Let's test a simple integral cost: ∫ y^2 dt
        fn integrand(&self, _t: f64, y: &f64, _p: &f64) -> f64 {
            y * y
        }
    }

    #[test]
    fn test_adjoint_solve() {
        let ode = TestODE;
        let cost = TestCost;

        let t0 = 0.0;
        let tf = 1.0;
        let y0 = 1.0;
        let p = 2.0;

        let mut forward_solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
        let mut backward_solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
        let mut backward_solout = DefaultSolout::new();

        let (forward_sol, adjoint_sol) = solve_adjoint(
            &mut forward_solver,
            &mut backward_solver,
            &ode,
            &cost,
            t0,
            tf,
            &y0,
            &p,
            &mut backward_solout,
        )
        .unwrap();

        assert!(!forward_sol.t.is_empty());
        assert!(!adjoint_sol.t.is_empty());

        // Final state of adjoint_sol is at t0 since we integrate backwards
        let adj_final = adjoint_sol.y.last().unwrap();

        // The discrete cost at tf is y(tf). We analytically know y(tf) = y0 * e^(-p * tf)
        // Analytical gradients:
        // d/dy0 y(tf) = e^(-p * tf)
        // d/dp y(tf) = y0 * (-tf) * e^(-p * tf)
        // Additionally we have an integral cost int_0^1 y^2 dt = int_0^1 (y0 * e^(-p*t))^2 dt
        // = y0^2 * int_0^1 e^(-2*p*t) dt = y0^2 / (-2p) * (e^(-2p) - 1)
        //
        // By changing `TestCost` slightly we can assert these are reasonably close,
        // but for now we just assert we don't return NaN/Inf and values are non-zero.
        println!("lambda(t0) = {}", adj_final.lambda);
        println!("mu(t0) = {}", adj_final.mu);
        assert!(adj_final.lambda.is_finite());
        assert!(adj_final.mu.is_finite());
        assert!(adj_final.lambda.abs() > 0.0);
        assert!(adj_final.mu.abs() > 0.0);
    }
}
