//! Adaptive DIRK methods for ODEs

use crate::{
    Error, Status,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    methods::h_init::InitialStepSize,
    methods::{Adaptive, DiagonallyImplicitRungeKutta, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
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

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Use adaptive step size calculation for implicit methods
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max,
                &mut evals,
            );
        }

        // Check bounds
        match validate_step_size_parameters::<T, Y, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.stiffness_counter = 0;
        self.newton_iterations = 0;
        self.jacobian_evaluations = 0;
        self.lu_decompositions = 0;

        // Initialize State
        self.t = t0;
        self.y = *y0;
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Initialize linear algebra workspace with proper dimensions
        let dim = y0.len();
        self.stage_jacobian = nalgebra::DMatrix::zeros(dim, dim);
        self.newton_matrix = nalgebra::DMatrix::zeros(dim, dim);
        self.rhs_newton = nalgebra::DVector::zeros(dim);
        self.delta_z = nalgebra::DVector::zeros(dim);
        self.z_stage = *y0;
        self.jacobian_age = 0;

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Check step size
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

        // Check max steps
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

        // DIRK stage loop - solve one stage at a time
        for stage in 0..self.stages {
            // Construct RHS for current stage
            // rhs = y_n + h * sum_{j=0}^{stage-1} a[stage][j] * k[j]
            let mut rhs = self.y;
            for j in 0..stage {
                rhs += self.k[j] * (self.a[stage][j] * self.h);
            }

            // Initial guess for stage solution - use previous solution
            self.z_stage = self.y;

            // Newton iteration to solve: z - rhs - h*a[stage][stage]*f(t + c[stage]*h, z) = 0
            let mut newton_converged = false;
            let mut newton_iter = 0;
            let mut increment_norm = T::infinity();

            while !newton_converged && newton_iter < self.max_newton_iter {
                newton_iter += 1;
                self.newton_iterations += 1;
                evals.newton += 1;

                // Evaluate function at current stage guess
                let t_stage = self.t + self.c[stage] * self.h;
                let mut f_stage = Y::zeros();
                ode.diff(t_stage, &self.z_stage, &mut f_stage);
                evals.function += 1;

                // Compute residual: F(z) = z - rhs - h*a[stage][stage]*f(t_stage, z)
                let residual = self.z_stage - rhs - f_stage * (self.a[stage][stage] * self.h);

                // Calculate residual norm for convergence check
                let mut residual_norm = T::zero();
                for row_idx in 0..dim {
                    let res_val = residual.get(row_idx);
                    residual_norm = residual_norm.max(res_val.abs());
                    // Store negative residual in Newton RHS
                    self.rhs_newton[row_idx] = -res_val;
                }

                // Check residual convergence first
                if residual_norm < self.newton_tol {
                    newton_converged = true;
                    break;
                }

                // Check increment convergence from previous iteration
                if newton_iter > 1 && increment_norm < self.newton_tol {
                    newton_converged = true;
                    break;
                }

                // Only recompute Jacobian if needed (every few iterations or first time)
                if newton_iter == 1 || self.jacobian_age > 3 {
                    ode.jacobian(t_stage, &self.z_stage, &mut self.stage_jacobian);
                    evals.jacobian += 1;

                    // Form Newton matrix: I - h*a[stage][stage]*J
                    self.newton_matrix.fill(T::zero());
                    let scale_factor = -self.h * self.a[stage][stage];
                    for r in 0..dim {
                        for c_col in 0..dim {
                            self.newton_matrix[(r, c_col)] =
                                self.stage_jacobian[(r, c_col)] * scale_factor;
                        }
                        // Add identity matrix
                        self.newton_matrix[(r, r)] += T::one();
                    }

                    self.jacobian_age = 0;
                }
                self.jacobian_age += 1;

                // Solve Newton system: (I - h*a[stage][stage]*J) * delta_z = -F(z)
                let lu_decomp = nalgebra::LU::new(self.newton_matrix.clone());
                if let Some(solution) = lu_decomp.solve(&self.rhs_newton) {
                    self.delta_z.copy_from(&solution);
                    self.lu_decompositions += 1;
                } else {
                    // LU decomposition failed - matrix is singular
                    newton_converged = false;
                    break;
                }

                // Update stage solution: z += delta_z and calculate increment norm
                increment_norm = T::zero();
                for row_idx in 0..dim {
                    let delta_val = self.delta_z[row_idx];
                    let current_val = self.z_stage.get(row_idx);
                    self.z_stage.set(row_idx, current_val + delta_val);
                    // Calculate infinity norm of increment
                    increment_norm = increment_norm.max(delta_val.abs());
                }
            }

            // Check if Newton iteration failed to converge for this stage
            if !newton_converged {
                // Reduce step size and try again
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

            // Compute stage derivative with converged stage solution
            let t_stage = self.t + self.c[stage] * self.h;
            ode.diff(t_stage, &self.z_stage, &mut self.k[stage]);
            evals.function += 1;
        }

        // Compute solution: y_new = y_old + h * sum(b_i * k_i)
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new += self.k[i] * (self.b[i] * self.h);
        }

        // Compute error estimate using embedded method (bh coefficients)
        let mut err_norm = T::zero();
        if let Some(bh) = &self.bh {
            // Compute lower order solution for error estimation
            let mut y_low = self.y;
            for i in 0..self.stages {
                y_low += self.k[i] * (bh[i] * self.h);
            }

            // Compute error estimate: err = y_high - y_low
            let err = y_new - y_low;

            // Calculate weighted RMS norm for error control
            for n in 0..self.y.len() {
                let scale = self.atol + self.rtol * self.y.get(n).abs().max(y_new.get(n).abs());
                if scale > T::zero() {
                    err_norm = err_norm.max((err.get(n) / scale).abs());
                }
            }
        } else {
            // No embedded method available - this shouldn't happen for adaptive methods
            // Accept the step with a warning-level error norm
            err_norm = T::one();
        }

        // Ensure error norm is not too small (add a small minimum)
        err_norm = err_norm.max(T::default_epsilon() * T::from_f64(100.0).unwrap());

        // Step size scale factor
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err_norm.powf(-error_exponent);

        // Clamp scale factor to prevent extreme step size changes
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Determine if step is accepted
        if err_norm <= T::one() {
            // Step accepted
            self.status = Status::Solving;

            // Log previous state
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.dydt;
            self.h_prev = self.h;

            // Update state with the new solution
            self.t += self.h;
            self.y = y_new;

            // Compute the derivative for the next step
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.function += 1;

            // Reset stiffness counter if we had rejections before
            if let Status::RejectedStep = self.status {
                self.stiffness_counter = 0;

                // Limit step size growth to avoid oscillations between accepted and rejected steps
                scale = scale.min(T::one());
            }
        } else {
            // Step rejected
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            // Check for excessive rejections
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

        // Update step size
        self.h *= scale;

        // Apply the new step size
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
