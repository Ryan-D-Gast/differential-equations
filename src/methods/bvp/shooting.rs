use crate::{
    bvp::bvp::BVP,
    error::Error,
    interpolate::Interpolation,
    ivp::IVP,
    linalg::{Matrix, lin_solve, lu_decomp},
    methods::bvp::BVPMethod,
    ode::{ODE, OrdinaryNumericalMethod},
    solution::Solution,
    traits::{Real, State},
};

/// Wrapper to adapt a BVP to an ODE.
struct BvpToOde<'a, EqType> {
    problem: &'a EqType,
}

impl<'a, EqType, T: Real, Y: State<T>> ODE<T, Y> for BvpToOde<'a, EqType>
where
    EqType: BVP<T, Y>,
{
    #[inline]
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        self.problem.diff(t, y, dydt);
    }
}

/// Shooting method for solving BVPs.
///
/// This method reduces a BVP to a sequence of IVPs. It iteratively
/// adjusts the initial state to satisfy the boundary conditions at tf.
pub struct ShootingMethod<M> {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub ode_solver: M,
}

impl<M> ShootingMethod<M> {
    pub fn new(ode_solver: M) -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            ode_solver,
        }
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl<M, T, Y> BVPMethod<T, Y> for ShootingMethod<M>
where
    T: Real,
    Y: State<T>,
    M: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y> + Clone,
{
    fn solve<EqType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y,
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: BVP<T, Y>,
    {
        let dim = y_guess.len();
        let mut y = y_guess.clone();
        let mut residual = y_guess.zeros_like();
        let mut jacobian = Matrix::<T>::zeros(dim, dim);
        let mut ip = vec![0; dim];

        let ode_system = BvpToOde { problem };

        for _iter in 0..self.max_iterations {
            // Solve ODE from t0 to tf with current initial guess
            let sol = IVP::ode(&ode_system, t0, tf, y.clone())
                .method(self.ode_solver.clone())
                .solve()?;

            let (_, y_f) = sol.last().unwrap();

            // Compute residual: res = g(y_a, y_b)
            problem.bound(&y, y_f, &mut residual);

            // Check convergence
            let mut max_res = T::zero();
            for i in 0..dim {
                let res_i = residual.get_component(i).abs();
                if res_i > max_res {
                    max_res = res_i;
                }
            }
            if max_res <= T::from_f64(self.tolerance).unwrap() {
                return Ok(sol);
            }

            // Compute Jacobian of the residual with respect to initial guess y
            // J = d(res)/dy_a = d g(y_a, y_b(y_a)) / dy_a
            let eps = T::default_epsilon().sqrt();
            for j in 0..dim {
                let mut y_perturbed = y.clone();
                let y_j = y.get_component(j);
                let perturbation = eps * y_j.abs().max(T::one());
                y_perturbed.set_component(j, y_j + perturbation);

                let sol_perturbed = IVP::ode(&ode_system, t0, tf, y_perturbed.clone())
                    .method(self.ode_solver.clone())
                    .solve()?;
                let (_, y_f_perturbed) = sol_perturbed.last().unwrap();

                let mut res_perturbed = residual.clone();
                problem.bound(&y_perturbed, y_f_perturbed, &mut res_perturbed);

                for i in 0..dim {
                    jacobian[(i, j)] =
                        (res_perturbed.get_component(i) - residual.get_component(i)) / perturbation;
                }
            }

            // Newton step: J * delta_y = -residual
            let mut negative_residual = y.zeros_like();
            for i in 0..dim {
                negative_residual.set_component(i, -residual.get_component(i));
            }

            lu_decomp(&mut jacobian, &mut ip)?;
            lin_solve(&jacobian, &mut negative_residual, &ip);

            // Update y
            for i in 0..dim {
                y.set_component(i, y.get_component(i) + negative_residual.get_component(i));
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
    use crate::prelude::*;

    struct HarmonicOscillatorBVP {
        omega_sq: f64,
        y_tf: f64,
    }

    impl BVP<f64, [f64; 2]> for HarmonicOscillatorBVP {
        fn diff(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            dydt[0] = y[1];
            dydt[1] = -self.omega_sq * y[0];
        }

        fn bound(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
            // y(0) = 0
            res[0] = y_a[0] - 0.0;
            // y(pi/2) = 1.0
            res[1] = y_b[0] - self.y_tf;
        }
    }

    #[test]
    fn test_shooting_method() {
        let bvp = HarmonicOscillatorBVP {
            omega_sq: 1.0,
            y_tf: 1.0, // sin(pi/2) = 1.0
        };

        // Guess: y(0) = 0.0 (satisfies res[0]), y'(0) = 0.5 (guess for the initial slope)
        let y_guess = [0.0, 0.5];
        let t0 = 0.0;
        let tf = std::f64::consts::PI / 2.0;

        let solver = ShootingMethod::new(ExplicitRungeKutta::dop853());

        let result = IVP::bvp(&bvp, t0, tf, y_guess)
            .method(solver)
            .solve()
            .unwrap();

        let (t_f, y_f) = result.last().unwrap();
        assert!((*t_f - tf).abs() < 1e-10);
        // Correct initial state should be [0.0, 1.0] to get sin(x)
        let (_, y_first) = result.iter().next().unwrap();
        assert!(y_first[0].abs() < 1e-5);
        assert!((y_first[1] - 1.0).abs() < 1e-5);

        // Final state should be [1.0, 0.0] since sin(pi/2)=1 and cos(pi/2)=0
        assert!((y_f[0] - 1.0).abs() < 1e-5);
        assert!(y_f[1].abs() < 1e-5);
    }
}
