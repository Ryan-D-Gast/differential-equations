//! Fixed Runge-Kutta methods for ODEs

use super::ImplicitRungeKutta;
use crate::{
    Error, Status,
    methods::{Ordinary, Fixed},
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{OrdinaryNumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};
use nalgebra::{DMatrix, DVector};

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
        self.jacobian_matrix = DMatrix::zeros(dim, dim);
        self.newton_matrix = DMatrix::zeros(newton_system_size, newton_system_size);
        self.rhs_newton = DVector::zeros(newton_system_size);
        self.delta_k_vec = DVector::zeros(newton_system_size);
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

        // Initial guess for stage values (predictor step)
        for i in 0..self.stages {
            self.y_stages[i] = self.y;
        }        
        
        // Newton iteration to solve the implicit system
        let mut newton_converged = false;
        let mut newton_iter = 0;
        let dim = self.y.len();        
        
        // Compute Jacobian using the ODE trait's jacobian method
        ode.jacobian(self.t, &self.y, &mut self.jacobian_matrix);
        evals.jac += 1;

        while !newton_converged && newton_iter < self.max_newton_iter {
            newton_iter += 1;
            self.newton_iterations += 1;

            // Compute stage derivatives with current stage values
            for i in 0..self.stages {
                ode.diff(self.t + self.c[i] * self.h, &self.y_stages[i], &mut self.k[i]);
            }
            evals.fcn += self.stages;

            // Compute residual phi(k) = k_i - f(t_n + c_i*h, Y_i)
            // where Y_i = y_n + h * sum(a_ij * k_j)
            for i in 0..self.stages {
                // First, compute Y_i from current k values
                self.y_stages[i] = self.y;
                for j in 0..self.stages {
                    self.y_stages[i] += self.k[j] * (self.a[i][j] * self.h);
                }
                
                // Evaluate f at Y_i
                let mut f_at_stage = V::zeros();
                ode.diff(self.t + self.c[i] * self.h, &self.y_stages[i], &mut f_at_stage);
                evals.fcn += 1;
                  
                // Compute residual: phi_i = k_i - f(t_n + c_i*h, Y_i)
                // Store -phi_i in rhs_newton for the Newton system
                for row_idx in 0..dim {
                    self.rhs_newton[i * dim + row_idx] = f_at_stage.get(row_idx) - self.k[i].get(row_idx);
                }
            }            
            
            // Form Newton matrix M = I - h * (A ⊗ J)
            // where A is the Runge-Kutta matrix and J is the Jacobian
            for i in 0..self.stages {
                for j in 0..self.stages {
                    let scale_factor = -self.h * self.a[i][j];
                    for r in 0..dim {
                        for c_col in 0..dim {
                            self.newton_matrix[(i * dim + r, j * dim + c_col)] = 
                                self.jacobian_matrix[(r, c_col)] * scale_factor;
                        }
                    }
                    
                    // Add identity for diagonal blocks
                    if i == j {
                        for d_idx in 0..dim {
                            self.newton_matrix[(i * dim + d_idx, j * dim + d_idx)] += T::one();
                        }
                    }
                }
            }            
            
            // Solve M * delta_k = rhs_newton
            let lu_decomp = nalgebra::LU::new(self.newton_matrix.clone());
            if let Some(solution) = lu_decomp.solve(&self.rhs_newton) {
                self.delta_k_vec.copy_from(&solution);
            } else {
                // LU decomposition failed - matrix is singular
                newton_converged = false;
                break;
            }
            
            // Update k values: k_i += delta_k_i
            let mut norm_delta_k_sq = T::zero();
            for i in 0..self.stages {
                for row_idx in 0..dim {
                    let delta_val = self.delta_k_vec[i * dim + row_idx];
                    let current_val = self.k[i].get(row_idx);
                    self.k[i].set(row_idx, current_val + delta_val);
                    norm_delta_k_sq += delta_val * delta_val;
                }
            }

            // Check convergence: ||delta_k|| < tolerance
            if norm_delta_k_sq < self.newton_tol * self.newton_tol {
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
        
        // Final evaluation of stage derivatives with converged values
        for i in 0..self.stages {
            // Recompute Y_i with final k values
            self.y_stages[i] = self.y;
            for j in 0..self.stages {
                self.y_stages[i] += self.k[j] * (self.a[i][j] * self.h);
            }
            ode.diff(self.t + self.c[i] * self.h, &self.y_stages[i], &mut self.k[i]);
        }
        evals.fcn += self.stages;        
        
        // Compute the solution using the b coefficients
        let mut y_new = self.y;
        for i in 0..self.stages {
            y_new += self.k[i] * (self.b[i] * self.h);
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