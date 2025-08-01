//! Fixed DIRK methods for ODEs

use crate::{
    Error, Status,
    methods::{DiagonallyImplicitRungeKutta, Ordinary, Fixed},
    stats::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{OrdinaryNumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> OrdinaryNumericalMethod<T, V, D> for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
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

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check max steps
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t, y: self.y
            });
            return Err(Error::MaxSteps {
                t: self.t, y: self.y
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
                rhs = rhs + self.k[j] * (self.a[stage][j] * self.h);
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
                let mut f_stage = V::zeros();
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
                            self.newton_matrix[(r, c_col)] = self.stage_jacobian[(r, c_col)] * scale_factor;
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
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
            }

            // Compute stage derivative with converged stage solution
            let t_stage = self.t + self.c[stage] * self.h;
            ode.diff(t_stage, &self.z_stage, &mut self.k[stage]);
            evals.function += 1;
        }

        // Compute the solution using the b coefficients: y_new = y_old + h * sum(b_i * k_i)
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new = y_new + self.k[i] * (self.b[i] * self.h);
        }

        // Step is always accepted for fixed step size methods
        self.status = Status::Solving;

        // Update state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;
        self.h_prev = self.h;
        
        self.t += self.h;
        self.y = y_new;
        
        // Compute the derivative for the next step
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        Ok(evals)
    }

    fn t(&self) -> T { self.t }
    fn y(&self) -> &V { &self.y }
    fn t_prev(&self) -> T { self.t_prev }
    fn y_prev(&self) -> &V { &self.y_prev }
    fn h(&self) -> T { self.h }
    fn set_h(&mut self, h: T) { self.h = h; }
    fn status(&self) -> &Status<T, V, D> { &self.status }
    fn set_status(&mut self, status: Status<T, V, D>) { self.status = status; }
}

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for DiagonallyImplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t
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
            t_interp
        );

        Ok(y_interp)
    }
}
