//! Fixed-step DIRK for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::Matrix,
    methods::{DiagonallyImplicitRungeKutta, Fixed, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y> for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, Y, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Validate step size bounds
        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Stats
        self.stiffness_counter = 0;
        self.newton_iterations = 0;
        self.jacobian_evaluations = 0;
        self.lu_decompositions = 0;

        // State
        self.t = t0;
        self.y = y0.clone();
        self.dydt = y0.zeros_like();
        self.y_prev = y0.clone();
        self.dydt_prev = y0.zeros_like();
        self.k = core::array::from_fn(|_| y0.zeros_like());
        self.z = y0.clone();
        self.rhs_newton = y0.zeros_like();
        self.delta_z = y0.zeros_like();
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        // Previous state
        self.t_prev = self.t;
        self.y_prev = self.y.clone();
        self.dydt_prev = self.dydt.clone();

        // Newton workspace
        let dim = y0.len();
        self.jacobian = Matrix::zeros(dim, dim);
        self.z = y0.clone();
        self.jacobian_age = 0;

        // Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Max steps guard
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y.clone(),
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y.clone(),
            });
        }
        self.steps += 1;

        let dim = self.y.len();

        // DIRK stage loop (sequential)
        for stage in 0..self.stages {
            // rhs = y_n + h Σ_{j<stage} a[stage][j] k[j]
            let mut rhs = self.y.clone();
            for j in 0..stage {
                rhs.add_scaled(self.a[stage][j] * self.h, &self.k[j]);
            }

            // Initial stage guess
            self.z = self.y.clone();

            // Newton: solve z - rhs - h*a_ii f(t_i, z) = 0
            let mut newton_converged = false;
            let mut newton_iter = 0;
            let mut increment_norm = T::infinity();

            while !newton_converged && newton_iter < self.max_newton_iter {
                newton_iter += 1;
                self.newton_iterations += 1;
                evals.newton += 1;

                // Evaluate f at stage guess
                let t_stage = self.t + self.c[stage] * self.h;
                let mut f_stage = self.y.zeros_like();
                ode.diff(t_stage, &self.z, &mut f_stage);
                evals.function += 1;

                // Residual F(z)
                let residual = self.z.plus_linear_combination(&[
                    (&rhs, -T::one()),
                    (&f_stage, -(self.a[stage][stage] * self.h)),
                ]);

                // Max-norm and RHS
                self.rhs_newton = residual.scaled(-T::one());
                let residual_norm = residual.max_norm();

                // Converged by residual
                if residual_norm < self.newton_tol {
                    newton_converged = true;
                    break;
                }

                // Converged by increment
                if newton_iter > 1 && increment_norm < self.newton_tol {
                    newton_converged = true;
                    break;
                }

                // Refresh Jacobian if needed
                if newton_iter == 1 || self.jacobian_age > 3 {
                    ode.jacobian(t_stage, &self.z, &mut self.jacobian);
                    evals.jacobian += 1;
                    self.jacobian_age = 0;

                    // Newton matrix: I - h*a_ii J
                    self.jacobian
                        .component_mul_mut(-self.h * self.a[stage][stage]);
                    self.jacobian += Matrix::identity(dim);
                }
                self.jacobian_age += 1;

                // Solve (I - h*a_ii J) Δz = -F(z) using in-place LU
                self.delta_z = self.jacobian.lin_solve(self.rhs_newton.clone()).unwrap();
                evals.solves += 1;

                // Update z and increment norm
                self.z.add_scaled(T::one(), &self.delta_z);
                increment_norm = self.delta_z.max_norm();
            }

            // Newton failed for this stage
            if !newton_converged {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y.clone(),
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y.clone(),
                });
            }

            // k_i from converged z
            let t_stage = self.t + self.c[stage] * self.h;
            ode.diff(t_stage, &self.z, &mut self.k[stage]);
            evals.function += 1;
        }

        // y_{n+1} = y_n + h Σ b_i k_i
        let mut y_new = self.y.clone();
        for i in 0..self.stages {
            y_new.add_scaled(self.b[i] * self.h, &self.k[i]);
        }

        // Fixed step: always accept
        self.status = Status::Solving;

        // Advance state
        self.t_prev = self.t;
        self.y_prev = self.y.clone();
        self.dydt_prev = self.dydt.clone();
        self.h_prev = self.h;

        self.t += self.h;
        self.y = y_new;

        // Next-step derivative
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }
    fn y(&self) -> &Y {
        &self.y
    }
    fn t_prev(&self) -> T {
        self.t_prev
    }
    fn y_prev(&self) -> &Y {
        &self.y_prev
    }
    fn h(&self) -> T {
        self.h
    }
    fn set_h(&mut self, h: T) {
        self.h = h;
    }
    fn status(&self) -> &Status<T, Y> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> Interpolation<T, Y>
    for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, Y, O, S, I>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // Otherwise use cubic Hermite interpolation
        let y_interp = cubic_hermite_interpolate(
            self.t_prev,
            self.t,
            &self.y_prev,
            &self.y,
            &self.dydt_prev,
            &self.dydt,
            t_interp,
        );

        Ok(y_interp)
    }
}
