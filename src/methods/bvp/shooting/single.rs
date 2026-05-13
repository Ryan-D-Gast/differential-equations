use crate::{
    bvp::Boundary,
    error::Error,
    interpolate::Interpolation,
    linalg::{Matrix, lin_solve, lu_decomp},
    methods::{ToleranceConfig, bvp::BVPMethod},
    ode::{ODE, OrdinaryNumericalMethod, solve_ode},
    solout::{DefaultSolout, Solout},
    solution::Solution,
    stats::{Evals, Steps},
    tolerance::Tolerance,
    traits::{Real, State},
};

/// Single-shooting method for ODE boundary value problems.
///
/// This method reduces a BVP to a sequence of IVPs and applies Newton iteration
/// to adjust the initial state until the endpoint boundary residual is small.
#[derive(Clone, Debug)]
pub struct SingleShooting<M> {
    max_iterations: usize,
    tolerance: f64,
    ode_solver: M,
}

impl<M> SingleShooting<M> {
    /// Create a single-shooting method from an ODE IVP solver.
    pub fn new(ode_solver: M) -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            ode_solver,
        }
    }

    /// Set the maximum number of Newton iterations.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the infinity-norm tolerance for the boundary residual.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl<M, T> ToleranceConfig<T> for SingleShooting<M>
where
    T: Real,
    M: ToleranceConfig<T>,
{
    fn rtol<V: Into<Tolerance<T>>>(mut self, rtol: V) -> Self {
        self.ode_solver = self.ode_solver.rtol(rtol);
        self
    }

    fn atol<V: Into<Tolerance<T>>>(mut self, atol: V) -> Self {
        self.ode_solver = self.ode_solver.atol(atol);
        self
    }
}

/// Wrapper to adapt a BVP definition to the ODE trait for internal IVP solves.
struct BvpToOde<'a, EqType: ?Sized> {
    problem: &'a EqType,
}

impl<EqType, T: Real, Y: State<T>> ODE<T, Y> for BvpToOde<'_, EqType>
where
    EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
{
    #[inline]
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        self.problem.diff(t, y, dydt);
    }
}

impl<M, T, Y> BVPMethod<T, Y> for SingleShooting<M>
where
    T: Real,
    Y: State<T>,
    M: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y> + Clone,
{
    fn solve<EqType, SoloutType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y,
        solout: &mut SoloutType,
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
        SoloutType: Solout<T, Y>,
    {
        let dim = y_guess.len();
        let mut y = y_guess.clone();
        let mut residual = y_guess.zeros_like();
        let mut jacobian = Matrix::<T>::zeros(dim, dim);
        let mut ip = vec![0; dim];
        let mut total_evals = Evals::new();
        let mut total_steps = Steps::new();
        let tolerance = T::from_f64(self.tolerance).ok_or_else(|| Error::BadInput {
            msg: "BVP shooting tolerance cannot be represented by scalar type.".to_string(),
        })?;

        let ode_system = BvpToOde { problem };

        for _ in 0..self.max_iterations {
            let mut trial_solver = self.ode_solver.clone();
            let mut trial_solout = DefaultSolout::new();
            let sol = solve_ode(
                &mut trial_solver,
                &ode_system,
                t0,
                tf,
                &y,
                &mut trial_solout,
            )?;
            total_evals += sol.evals;
            total_steps += sol.steps;

            let (_, y_f) = sol.last().map_err(|err| Error::BadInput {
                msg: format!("Internal IVP solve returned an empty solution: {err}"),
            })?;

            problem.boundary(&y, y_f, &mut residual);

            if residual.max_norm() <= tolerance {
                let mut final_solver = self.ode_solver.clone();
                let mut solution = solve_ode(&mut final_solver, &ode_system, t0, tf, &y, solout)?;
                solution.evals += total_evals;
                solution.steps += total_steps;
                return Ok(solution);
            }

            let eps = T::default_epsilon().sqrt();
            for j in 0..dim {
                let mut y_perturbed = y.clone();
                let y_j = y.get_component(j);
                let perturbation = eps * y_j.abs().max(T::one());
                y_perturbed.set_component(j, y_j + perturbation);

                let mut perturbed_solver = self.ode_solver.clone();
                let mut perturbed_solout = DefaultSolout::new();
                let sol_perturbed = solve_ode(
                    &mut perturbed_solver,
                    &ode_system,
                    t0,
                    tf,
                    &y_perturbed,
                    &mut perturbed_solout,
                )?;
                total_evals += sol_perturbed.evals;
                total_steps += sol_perturbed.steps;
                let (_, y_f_perturbed) = sol_perturbed.last().map_err(|err| Error::BadInput {
                    msg: format!("Internal perturbed IVP solve returned an empty solution: {err}"),
                })?;

                let mut res_perturbed = residual.clone();
                problem.boundary(&y_perturbed, y_f_perturbed, &mut res_perturbed);
                total_evals.jacobian += 1;

                for i in 0..dim {
                    jacobian[(i, j)] =
                        (res_perturbed.get_component(i) - residual.get_component(i)) / perturbation;
                }
            }

            let mut step = y.zeros_like();
            for i in 0..dim {
                step.set_component(i, -residual.get_component(i));
            }

            lu_decomp(&mut jacobian, &mut ip).map_err(|err| Error::LinearAlgebra {
                t: t0,
                y: y.clone(),
                msg: err.to_string(),
            })?;
            lin_solve(&jacobian, &mut step, &ip);
            total_evals.newton += 1;
            total_evals.decompositions += 1;
            total_evals.solves += 1;

            for i in 0..dim {
                y.set_component(i, y.get_component(i) + step.get_component(i));
            }
        }

        Err(Error::MaxSteps {
            t: t0,
            y: y_guess.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bvp::BVP,
        methods::{ExplicitRungeKutta, bvp::Shooting},
    };

    struct HarmonicOscillatorBvp {
        target: f64,
    }

    impl ODE<f64, [f64; 2]> for HarmonicOscillatorBvp {
        fn diff(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            dydt[0] = y[1];
            dydt[1] = -y[0];
        }
    }

    impl Boundary<f64, [f64; 2]> for HarmonicOscillatorBvp {
        fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
            res[0] = y_a[0];
            res[1] = y_b[0] - self.target;
        }
    }

    #[test]
    fn shooting_solves_harmonic_oscillator_with_trait_api() {
        let problem = HarmonicOscillatorBvp { target: 1.0 };
        let method = Shooting::single(ExplicitRungeKutta::dop853());

        let result = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
            .method(method)
            .solve()
            .expect("BVP solve should converge");

        let (_, y_initial) = result.iter().next().expect("solution has an initial point");
        let (_, y_final) = result.last().expect("solution has a final point");

        assert!(y_initial[0].abs() < 1e-5);
        assert!((y_initial[1] - 1.0).abs() < 1e-5);
        assert!((y_final[0] - 1.0).abs() < 1e-5);
        assert!(y_final[1].abs() < 1e-5);
    }

    #[test]
    fn shooting_solves_harmonic_oscillator_with_closure_api() {
        let method = Shooting::single(ExplicitRungeKutta::dop853());

        let result = BVP::ode_from_fn(
            |_t, y: &[f64; 2], dydt: &mut [f64; 2]| {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            },
            |y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]| {
                res[0] = y_a[0];
                res[1] = y_b[0] - 1.0;
            },
            0.0,
            std::f64::consts::FRAC_PI_2,
            [0.0, 0.5],
        )
        .method(method)
        .solve()
        .expect("BVP solve should converge");

        let (_, y_initial) = result.iter().next().expect("solution has an initial point");
        let (_, y_final) = result.last().expect("solution has a final point");

        assert!((y_initial[1] - 1.0).abs() < 1e-5);
        assert!((y_final[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shooting_supports_t_eval_output_for_final_trajectory() {
        let problem = HarmonicOscillatorBvp { target: 1.0 };
        let method = Shooting::single(ExplicitRungeKutta::dop853());
        let points = [
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
        ];

        let result = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
            .t_eval(points)
            .method(method)
            .solve()
            .expect("BVP solve should converge with t_eval output");

        assert_eq!(result.t, points);
        assert_eq!(result.y.len(), points.len());
        assert!((result.y[0][1] - 1.0).abs() < 1e-5);
        assert!((result.y[2][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn shooting_reports_internal_ivp_and_newton_statistics() {
        let problem = HarmonicOscillatorBvp { target: 1.0 };
        let method = Shooting::single(ExplicitRungeKutta::dop853().rtol(1e-10).atol(1e-12));

        let result = BVP::ode(&problem, 0.0, std::f64::consts::FRAC_PI_2, [0.0, 0.5])
            .method(method)
            .solve()
            .expect("BVP solve should converge");

        assert!(result.evals.function > 0);
        assert!(result.evals.jacobian > 0);
        assert!(result.evals.newton > 0);
        assert_eq!(result.evals.decompositions, result.evals.newton);
        assert_eq!(result.evals.solves, result.evals.newton);
        assert!(result.steps.total() > 0);
    }
}
