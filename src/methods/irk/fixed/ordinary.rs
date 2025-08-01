//! Fixed Runge-Kutta methods for ODEs

use crate::{
    Error, Status,
    methods::{ImplicitRungeKutta, Ordinary, Fixed},
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{OrdinaryNumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> OrdinaryNumericalMethod<T, V, D> for ImplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();        

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            // Set the fixed step size
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
        evals.fcn += 1;

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Initialize linear algebra workspace with proper dimensions
        let dim = y0.len();
        let newton_system_size = self.stages * dim;
        self.stage_jacobians = core::array::from_fn(|_| nalgebra::DMatrix::zeros(dim, dim));
        self.newton_matrix = nalgebra::DMatrix::zeros(newton_system_size, newton_system_size);
        self.rhs_newton = nalgebra::DVector::zeros(newton_system_size);
        self.delta_k_vec = nalgebra::DVector::zeros(newton_system_size);
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

        // Initial guess for stage values - use previous solution for all stages
        let dim = self.y.len();
        for i in 0..self.stages {
            self.y_stages[i] = self.y;
        }        
        
        // Newton iteration to solve the implicit system
        // We solve F(z) = z - y_n - h*A*f(z) = 0 where z = [z1; z2; ...; zs]
        let mut newton_converged = false;
        let mut newton_iter = 0;

        while !newton_converged && newton_iter < self.max_newton_iter {
            newton_iter += 1;
            self.newton_iterations += 1;

            // Evaluate f at current stage guesses and compute residual
            for i in 0..self.stages {
                ode.diff(self.t + self.c[i] * self.h, &self.y_stages[i], &mut self.k[i]);
            }
            evals.fcn += self.stages;

            // Compute residual F(z) = z - y_n - h*sum(A*f)
            for i in 0..self.stages {
                // Start with z_i - y_n
                let mut residual = self.y_stages[i] - self.y;
                
                // Subtract h*sum(a_ij * f_j)
                for j in 0..self.stages {
                    residual = residual - self.k[j] * (self.a[i][j] * self.h);
                }
                
                // Store residual in Newton RHS (negative for solving delta_z)
                for row_idx in 0..dim {
                    self.rhs_newton[i * dim + row_idx] = -residual.get(row_idx);
                }
            }

            // Only recompute Jacobian if needed (every few iterations or first time)
            if newton_iter == 1 || self.jacobian_age > 3 {
                // Evaluate Jacobian at each stage point like MATLAB does
                // Use pre-allocated stage_jacobians array to avoid dynamic allocation
                for i in 0..self.stages {
                    ode.jacobian(self.t + self.c[i] * self.h, &self.y_stages[i], &mut self.stage_jacobians[i]);
                    evals.jac += 1;
                }
                
                // Form Newton matrix: I - h * (A ⊗ J) using stage-specific Jacobians
                self.newton_matrix.fill(T::zero());
                for i in 0..self.stages {
                    for j in 0..self.stages {
                        let scale_factor = -self.h * self.a[i][j];
                        // Use the Jacobian from stage j (like MATLAB: A(i,j)*J{j})
                        for r in 0..dim {
                            for c_col in 0..dim {
                                self.newton_matrix[(i * dim + r, j * dim + c_col)] = 
                                    self.stage_jacobians[j][(r, c_col)] * scale_factor;
                            }
                        }
                    }
                    
                    // Add identity matrix for diagonal blocks
                    for d_idx in 0..dim {
                        self.newton_matrix[(i * dim + d_idx, i * dim + d_idx)] += T::one();
                    }
                }
                
                self.jacobian_age = 0;
            }
            self.jacobian_age += 1;
            
            // Solve Newton system: (I - h*A⊗J) * delta_z = -F(z)
            let lu_decomp = nalgebra::LU::new(self.newton_matrix.clone());
            if let Some(solution) = lu_decomp.solve(&self.rhs_newton) {
                self.delta_k_vec.copy_from(&solution);
                self.lu_decompositions += 1;
            } else {
                // LU decomposition failed - matrix is singular
                newton_converged = false;
                break;
            }
            
            // Update stage values: z_i += delta_z_i
            let mut norm_delta_sq = T::zero();
            for i in 0..self.stages {
                for row_idx in 0..dim {
                    let delta_val = self.delta_k_vec[i * dim + row_idx];
                    let current_val = self.y_stages[i].get(row_idx);
                    self.y_stages[i].set(row_idx, current_val + delta_val);
                    norm_delta_sq += delta_val * delta_val;
                }
            }

            // Check convergence: ||delta_z|| < tolerance
            if norm_delta_sq < self.newton_tol * self.newton_tol {
                newton_converged = true;
            }
        }        
          
        // Check if Newton iteration failed to converge
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
        
        // Compute final stage derivatives with converged stage values
        for i in 0..self.stages {
            ode.diff(self.t + self.c[i] * self.h, &self.y_stages[i], &mut self.k[i]);
        }
        evals.fcn += self.stages;        
        
        // Compute the solution using the b coefficients: y_new = y_old + h * sum(b_i * f_i)
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new = y_new + self.k[i] * (self.b[i] * self.h);
        }

        // Step is always accepted for fixed step size methods
        self.status = Status::Solving;

        // Log previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;
        self.h_prev = self.h;

        // Update state
        self.t += self.h;
        self.y = y_new;

        // Compute the derivative for the next step
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.fcn += 1;
        
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

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ImplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {
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