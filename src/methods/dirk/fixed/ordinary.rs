//! Fixed-step DIRK for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::Matrix,
    methods::{DiagonallyImplicitRungeKutta, Fixed, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y, D>
    for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, Y, D, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Validate step size bounds
        match validate_step_size_parameters::<T, Y, D>(self.h0, self.h_min, self.h_max, t0, tf) {
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
        self.y = *y0;
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        // Previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Newton workspace
        let dim = y0.len();
        self.jacobian = Matrix::zeros(dim, dim);
        self.z = *y0;
        self.jacobian_age = 0;

        // Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Max steps guard
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
        }
        self.steps += 1;

        let dim = self.y.len();

        // DIRK stage loop (sequential)
        for stage in 0..self.stages {
            // rhs = y_n + h Σ_{j<stage} a[stage][j] k[j]
            let mut rhs = self.y;
            for j in 0..stage {
                rhs += self.k[j] * (self.a[stage][j] * self.h);
            }

            // Initial stage guess
            self.z = self.y;

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
                let mut f_stage = Y::zeros();
                ode.diff(t_stage, &self.z, &mut f_stage);
                evals.function += 1;

                // Residual F(z)
                let residual = self.z - rhs - f_stage * (self.a[stage][stage] * self.h);

                // Max-norm and RHS
                let mut residual_norm = T::zero();
                self.rhs_newton = -residual;
                for i in 0..dim {
                    residual_norm = residual_norm.max(residual.get(i).abs());
                }

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
                self.delta_z = self.jacobian.lin_solve(self.rhs_newton).unwrap();
                self.lu_decompositions += 1;

                // Update z and increment norm
                increment_norm = T::zero();
                self.z = self.z + self.delta_z;
                for row_idx in 0..dim {
                    // Calculate infinity norm of increment
                    increment_norm = increment_norm.max(self.delta_z.get(row_idx).abs());
                }
            }

            // Newton failed for this stage
            if !newton_converged {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
            }

            // k_i from converged z
            let t_stage = self.t + self.c[stage] * self.h;
            ode.diff(t_stage, &self.z, &mut self.k[stage]);
            evals.function += 1;
        }

        // y_{n+1} = y_n + h Σ b_i k_i
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new += self.k[i] * (self.b[i] * self.h);
        }

        // Fixed step: always accept
        self.status = Status::Solving;

        // Advance state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;
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
    fn status(&self) -> &Status<T, Y, D> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y, D>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Interpolation<T, Y> for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, Y, D, O, S, I>
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
