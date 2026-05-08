//! Adaptive DIRK for ODEs
use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    linalg::Matrix,
    methods::h_init::InitialStepSize,
    methods::{Adaptive, DiagonallyImplicitRungeKutta, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y>
    for DiagonallyImplicitRungeKutta<Ordinary, Adaptive, T, Y, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Compute h0 if not set
        if self.h0 == T::zero() {
            // Implicit initial step size heuristic
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, self.order, &self.rtol, &self.atol, self.h_min, self.h_max,
                &mut evals,
            );
        }

        // Validate step size bounds
        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = (self.filter)(h0),
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

        // Step size guard
        if self.h.abs() < self.h_prev.abs() * T::from_f64(1e-14).unwrap() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y.clone(),
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y.clone(),
            });
        }

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
                let mut residual_values = vec![T::zero(); dim];
                residual.write_to_slice(&mut residual_values);
                let residual_norm = residual_values
                    .iter()
                    .fold(T::zero(), |norm, value| norm.max(value.abs()));

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
                self.lu_decompositions += 1;

                // Update z and increment norm
                self.z.add_scaled(T::one(), &self.delta_z);
                let mut delta_values = vec![T::zero(); dim];
                self.delta_z.write_to_slice(&mut delta_values);
                increment_norm = delta_values
                    .iter()
                    .fold(T::zero(), |norm, value| norm.max(value.abs()));
            }

            // Newton failed for this stage
            if !newton_converged {
                // Reduce h and retry later
                self.h *= T::from_f64(0.25).unwrap();
                self.h = constrain_step_size(self.h, self.h_min, self.h_max);
                self.h = (self.filter)(self.h);
                self.status = Status::RejectedStep;
                self.stiffness_counter += 1;

                if self.stiffness_counter >= self.max_rejects {
                    self.status = Status::Error(Error::Stiffness {
                        t: self.t,
                        y: self.y.clone(),
                    });
                    return Err(Error::Stiffness {
                        t: self.t,
                        y: self.y.clone(),
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
        let mut y_new = self.y.clone();
        for i in 0..self.stages {
            y_new.add_scaled(self.b[i] * self.h, &self.k[i]);
        }

        // Embedded error estimate (bh)
        let mut err_norm = T::zero();
        let bh = &self.bh.unwrap();

        // Lower-order solution
        let mut y_low = self.y.clone();
        for i in 0..self.stages {
            y_low.add_scaled(bh[i] * self.h, &self.k[i]);
        }

        // Weighted max-norm
        let dim = self.y.len();
        let mut y_values = vec![T::zero(); dim];
        let mut y_new_values = vec![T::zero(); dim];
        let mut y_low_values = vec![T::zero(); dim];
        self.y.write_to_slice(&mut y_values);
        y_new.write_to_slice(&mut y_new_values);
        y_low.write_to_slice(&mut y_low_values);
        for i in 0..dim {
            let scale = self.atol[i] + self.rtol[i] * y_values[i].abs().max(y_new_values[i].abs());
            if scale > T::zero() {
                err_norm = err_norm.max(((y_new_values[i] - y_low_values[i]) / scale).abs());
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
            self.y_prev = self.y.clone();
            self.dydt_prev = self.dydt.clone();
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
                    y: self.y.clone(),
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y.clone(),
                });
            }
        }

        // Update h
        self.h *= scale;

        // Constrain h
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);

        // Apply step size filter
        self.h = (self.filter)(self.h);

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
        self.h = (self.filter)(h);
    }
    fn status(&self) -> &Status<T, Y> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> Interpolation<T, Y>
    for DiagonallyImplicitRungeKutta<Ordinary, Adaptive, T, Y, O, S, I>
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
