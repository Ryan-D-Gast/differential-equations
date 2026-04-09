//! Fixed-step IRK for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::Matrix,
    methods::{Fixed, ImplicitRungeKutta, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y> for ImplicitRungeKutta<Ordinary, Fixed, T, Y, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Validate step size bounds
        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            // Set the fixed step size
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

        // Linear algebra workspace
        let dim = y0.len();
        let newton_system_size = self.stages * dim;
        self.stage_jacobians = core::array::from_fn(|_| Matrix::zeros(dim, dim));
        self.newton_matrix = Matrix::zeros(newton_system_size, newton_system_size);
        self.rhs_newton = vec![T::zero(); newton_system_size];
        self.delta_k_vec = vec![T::zero(); newton_system_size];
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
                y: self.y,
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
        }
        self.steps += 1;

        // Initial stage guesses: copy current state
        let dim = self.y.len();
        for i in 0..self.stages {
            self.z[i] = self.y;
        }

        // Newton solve for F(z) = z - y_n - h*A*f(z) = 0
        let mut newton_converged = false;
        let mut newton_iter = 0;

        // Track increment norm
        let mut increment_norm = T::infinity();

        while !newton_converged && newton_iter < self.max_newton_iter {
            newton_iter += 1;
            self.newton_iterations += 1;
            evals.newton += 1;

            // Evaluate f at stage guesses
            for i in 0..self.stages {
                ode.diff(self.t + self.c[i] * self.h, &self.z[i], &mut self.k[i]);
            }
            evals.function += self.stages;

            // Residual and max-norm
            let mut residual_norm = T::zero();
            for i in 0..self.stages {
                // Start with z_i - y_n
                let mut residual = self.z[i] - self.y;

                // Subtract h*sum(a_ij * f_j)
                for j in 0..self.stages {
                    residual = residual - self.k[j] * (self.a[i][j] * self.h);
                }

                // Infinity norm and RHS
                for row_idx in 0..dim {
                    let res_val = residual.get(row_idx);
                    residual_norm = residual_norm.max(res_val.abs());
                    // Store residual in Newton RHS (negative for solving delta_z)
                    self.rhs_newton[i * dim + row_idx] = -res_val;
                }
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

            // Refresh Jacobians if needed
            if newton_iter == 1 || self.jacobian_age > 3 {
                // Stage Jacobians
                for i in 0..self.stages {
                    ode.jacobian(
                        self.t + self.c[i] * self.h,
                        &self.z[i],
                        &mut self.stage_jacobians[i],
                    );
                    evals.jacobian += 1;
                }

                // Build Newton matrix: I - h*(A ⊗ J)
                let nsys = self.stages * dim;
                let mut nm = Matrix::zeros(nsys, nsys);
                for i in 0..self.stages {
                    for j in 0..self.stages {
                        let scale_factor = -self.h * self.a[i][j];
                        // Use J from stage j
                        for r in 0..dim {
                            for c_col in 0..dim {
                                nm[(i * dim + r, j * dim + c_col)] =
                                    self.stage_jacobians[j][(r, c_col)] * scale_factor;
                            }
                        }
                    }

                    // Add identity per block
                    for d_idx in 0..dim {
                        let idx = i * dim + d_idx;
                        nm[(idx, idx)] += T::one();
                    }
                }
                self.newton_matrix = nm;

                self.jacobian_age = 0;
            }
            self.jacobian_age += 1;

            // Solve (I - h*A⊗J) Δz = -F(z) using in-place LU on our matrix
            let mut rhs = self.rhs_newton.clone();
            self.newton_matrix.lin_solve_mut(&mut rhs[..]);
            for i in 0..self.delta_k_vec.len() {
                self.delta_k_vec[i] = rhs[i];
            }
            self.lu_decompositions += 1;

            // Update z_i and increment norm
            increment_norm = T::zero();
            for i in 0..self.stages {
                for row_idx in 0..dim {
                    let delta_val = self.delta_k_vec[i * dim + row_idx];
                    let current_val = self.z[i].get(row_idx);
                    self.z[i].set(row_idx, current_val + delta_val);
                    // Calculate infinity norm of increment
                    increment_norm = increment_norm.max(delta_val.abs());
                }
            }

            // Next loop will re-check
        }

        // Newton failed to converge
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

        // Final stage derivatives
        for i in 0..self.stages {
            ode.diff(self.t + self.c[i] * self.h, &self.z[i], &mut self.k[i]);
        }
        evals.function += self.stages;

        // y_{n+1} = y_n + h Σ b_i f_i
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new += self.k[i] * (self.b[i] * self.h);
        }

        // Fixed step: always accept
        self.status = Status::Solving;

        // Log previous
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;
        self.h_prev = self.h;

        // Advance state
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
    for ImplicitRungeKutta<Ordinary, Fixed, T, Y, O, S, I>
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

        // Use cubic Hermite interpolation
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
