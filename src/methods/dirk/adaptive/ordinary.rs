//! Adaptive DIRK for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::SquareMatrix,
    methods::h_init::InitialStepSize,
    methods::{Adaptive, DiagonallyImplicitRungeKutta, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y, D>
    for DiagonallyImplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
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

        // Newton workspace
        let dim = y0.len();
        self.jacobian = SquareMatrix::zeros(dim);
        self.newton_matrix = SquareMatrix::zeros(dim);
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

                    // Newton matrix: I - h*a_ii J
                    self.newton_matrix = SquareMatrix::zeros(dim);
                    let scale_factor = -self.h * self.a[stage][stage];
                    for r in 0..dim {
                        for c_col in 0..dim {
                            self.newton_matrix[(r, c_col)] =
                                self.jacobian[(r, c_col)] * scale_factor;
                        }
                        // Add identity
                        self.newton_matrix[(r, r)] = self.newton_matrix[(r, r)] + T::one();
                    }

                    self.jacobian_age = 0;
                }
                self.jacobian_age += 1;

                // Solve (I - h*a_ii J) Δz = -F(z) using in-place LU
                self.delta_z = self.newton_matrix.lin_solve(self.rhs_newton);
                self.lu_decompositions += 1;

                // Update z and increment norm
                increment_norm = T::zero();
                for row_idx in 0..dim {
                    let delta_val = self.delta_z.get(row_idx);
                    let current_val = self.z.get(row_idx);
                    self.z.set(row_idx, current_val + delta_val);
                    // Calculate infinity norm of increment
                    increment_norm = increment_norm.max(delta_val.abs());
                }
            }

            // Newton failed for this stage
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
    Interpolation<T, Y> for DiagonallyImplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
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
