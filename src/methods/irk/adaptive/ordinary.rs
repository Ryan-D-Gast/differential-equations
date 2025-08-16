//! Adaptive IRK for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::SquareMatrix,
    methods::h_init::InitialStepSize,
    methods::{Adaptive, ImplicitRungeKutta, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y, D> for ImplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Compute h0 if not set
        if self.h0 == T::zero() {
            // Implicit initial step size heuristic
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max,
                &mut evals,
            );
        }

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

        // Linear algebra workspace
        let dim = y0.len();
        let newton_system_size = self.stages * dim;
        self.stage_jacobians = core::array::from_fn(|_| SquareMatrix::zeros(dim));
        self.newton_matrix = SquareMatrix::zeros(newton_system_size);
        // Use State<T> storage for RHS and solution vectors
        self.rhs_newton = vec![T::zero(); newton_system_size];
        self.delta_k_vec = vec![T::zero(); newton_system_size];
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

        // Step size guard
        if self.h.abs() < self.h_prev.abs() * T::from_f64(1e-14).unwrap() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y,
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y,
            });
        }

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
                // Zero the block Newton matrix (ensure Full storage)
                let nsys = self.stages * dim;
                let mut nm = SquareMatrix::zeros(nsys);
                // Fill blocks
                for i in 0..self.stages {
                    for j in 0..self.stages {
                        let scale = -self.h * self.a[i][j];
                        for r in 0..dim {
                            for c_col in 0..dim {
                                nm[(i * dim + r, j * dim + c_col)] =
                                    self.stage_jacobians[j][(r, c_col)] * scale;
                            }
                        }
                    }
                    // Add identity on block diagonal
                    for d_idx in 0..dim {
                        let idx = i * dim + d_idx;
                        nm[(idx, idx)] = nm[(idx, idx)] + T::one();
                    }
                }
                self.newton_matrix = nm;

                self.jacobian_age = 0;
            }
            self.jacobian_age += 1;

            // Solve (I - h*A⊗J) Δz = -F(z) using our LU in-place over a flat slice
            let mut rhs = self.rhs_newton.clone();
            self.newton_matrix.lin_solve_in_place(&mut rhs[..]);
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
            // Reduce h and retry later
            self.h *= T::from_f64(0.25).unwrap();
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            if self.stiffness_counter >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
            }
            return Ok(evals);
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

        // Embedded error estimate (bh)
        let mut err_norm = T::zero();
        let bh = &self.bh.unwrap();

        // Lower-order solution
        let mut y_low = self.y;
        for i in 0..self.stages {
            y_low += self.k[i] * (bh[i] * self.h);
        }

        // err = y_high - y_low
        let err = y_new - y_low;

        // Weighted max-norm
        for n in 0..self.y.len() {
            let scale = self.atol + self.rtol * self.y.get(n).abs().max(y_new.get(n).abs());
            if scale > T::zero() {
                err_norm = err_norm.max((err.get(n) / scale).abs());
            }
        }

        // Avoid vanishing error
        err_norm = err_norm.max(T::default_epsilon() * T::from_f64(100.0).unwrap());

        // Step scale factor
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err_norm.powf(-error_exponent);

        // Clamp scale factor
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Accept/reject
        if err_norm <= T::one() {
            // Accepted
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

            // If we were rejecting, limit growth
            if let Status::RejectedStep = self.status {
                self.stiffness_counter = 0;

                // Avoid oscillations
                scale = scale.min(T::one());
            }
        } else {
            // Rejected
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            // Too many rejections
            if self.stiffness_counter >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
            }
        }

        // Update h
        self.h *= scale;

        // Constrain h
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);

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
    Interpolation<T, Y> for ImplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
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
