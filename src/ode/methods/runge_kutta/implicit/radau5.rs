//! Adaptive step size implicit Radau IIA 5th order method for solving ODEs.

use crate::{
    Error, Status,
    alias::Evals,
    interpolate::Interpolation,
    ode::{NumericalMethod, ODE, methods::h_init},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

// Constants for Radau IIA 3-stage, 5th order method
const SQRT6: f64 = 2.449489743; // sqrt(6.0)

// Time points c_i
const C0: f64 = (4.0 - SQRT6) / 10.0;
const C1: f64 = (4.0 + SQRT6) / 10.0;
const C2: f64 = 1.0;

// Butcher tableau A matrix coefficients
const A11: f64 = (88.0 - 7.0 * SQRT6) / 360.0;
const A12: f64 = (296.0 - 169.0 * SQRT6) / 1800.0;
const A13: f64 = (-2.0 + 3.0 * SQRT6) / 225.0;
const A21: f64 = (296.0 + 169.0 * SQRT6) / 1800.0;
const A22: f64 = (88.0 + 7.0 * SQRT6) / 360.0;
const A23: f64 = (-2.0 - 3.0 * SQRT6) / 225.0;
const A31: f64 = (16.0 - SQRT6) / 36.0;
const A32: f64 = (16.0 + SQRT6) / 36.0;
const A33: f64 = 1.0 / 9.0;

// Primary weights b_i
const B0: f64 = (16.0 - SQRT6) / 36.0;
const B1: f64 = (16.0 + SQRT6) / 36.0;
const B2: f64 = 1.0 / 9.0;

// Embedded method weights b_hat_i for error estimation
const BHAT0: f64 = (16.0 - SQRT6) / 36.0 - 0.01;
const BHAT1: f64 = (16.0 + SQRT6) / 36.0 - 0.01;
const BHAT2: f64 = 1.0 / 9.0 + 0.02;

// TODO: Add matrix coefficients and different step size selection strategies used
// in the Fortran implementation. Current version only implements core and dense output
// functionality with newton iteration and jacobian calls.

/// Radau IIA method of order 5 (3 stages).
///
/// This is a 3-stage, 5th order implicit Runge-Kutta method, A-stable, L-stable,
/// suitable for stiff problems. It uses Newton iteration, embedded error estimation,
/// and specialized Radau dense output for accurate interpolation.
/// 
/// Note currently this uses the same constants as the Radau IIA 3-stage method.
/// Unlike Fortran implemnentation no matrix, ti, coefficients are used and thus
/// this is a simplified version. 
/// 
pub struct Radau5<T: Real, V: State<T>, D: CallBackData> {
    // Initial Step Size
    pub h0: T,
    // Current Step Size
    h: T,
    // Previous step size (used for interpolation)
    h_prev_step: T,

    // Current State (t_n, y_n)
    t: T,
    y: V,
    dydt: V, // Derivative f(t_n, y_n)

    // Previous State (t_{n-1}, y_{n-1})
    t_prev: T,
    y_prev: V,
    dydt_prev: V, // Derivative f(t_{n-1}, y_{n-1})

    // Stage derivatives k_i = f(t_n + c_i*h, Y_i)
    k: [V; 3],
    // Stage values Y_i = y_n + h * sum(a_ij * k_j)
    y_stage: [V; 3],
    // f(t_stage, y_stage) during Newton iteration
    f_at_stages: [V; 3],

    // Butcher tableau coefficients (typed)
    a: [[T; 3]; 3],
    b_higher: [T; 3], // Primary weights b
    b_lower: [T; 3],  // Secondary weights b_hat for error estimation
    c: [T; 3],        // Time points

    // Dense output coefficients for polynomial interpolation
    cont: [V; 4],

    // --- Adaptive Step Settings ---
    pub rtol: T,
    pub atol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub max_rejects: usize,
    pub safety_factor: T,
    pub min_scale: T,
    pub max_scale: T,

    // --- Implicit Solver Settings ---
    pub max_iter: usize, // Max iterations for Newton solver
    pub tol: T,          // Tolerance for Newton solver convergence
    fd_epsilon_sqrt: T, // Stores sqrt(machine_epsilon) for FD

    // Iteration tracking & Status
    reject: bool,
    n_stiff: usize,
    steps: usize,
    status: Status<T, V, D>,

    // --- Jacobian and Newton Solver Data ---
    jacobian_matrix: nalgebra::DMatrix<T>, // Jacobian of f: J(t,y)
    newton_matrix: nalgebra::DMatrix<T>,   // Matrix for Newton system (M)
    rhs_newton: nalgebra::DVector<T>,      // RHS vector for Newton system (-phi)
    delta_k_vec: nalgebra::DVector<T>,     // Solution of Newton system (delta_k)
}

impl<T: Real, V: State<T>, D: CallBackData> Default for Radau5<T, V, D> {
    fn default() -> Self {
        // Convert Butcher tableau f64 constants to type T
        let a_coeffs: [[f64; 3]; 3] = [
            [A11, A12, A13],
            [A21, A22, A23],
            [A31, A32, A33]
        ];
        let b_coeffs: [f64; 3] = [B0, B1, B2];
        let b_hat_coeffs: [f64; 3] = [BHAT0, BHAT1, BHAT2];
        let c_coeffs: [f64; 3] = [C0, C1, C2];

        let a_t: [[T; 3]; 3] = a_coeffs.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b_higher_t: [T; 3] = b_coeffs.map(|x| T::from_f64(x).unwrap());
        let b_lower_t: [T; 3] = b_hat_coeffs.map(|x| T::from_f64(x).unwrap());
        let c_t: [T; 3] = c_coeffs.map(|x| T::from_f64(x).unwrap());

        Radau5 {
            h0: T::zero(), 
            h: T::zero(),
            h_prev_step: T::zero(),
            t: T::zero(),
            y: V::zeros(),
            dydt: V::zeros(),
            t_prev: T::zero(),
            y_prev: V::zeros(),
            dydt_prev: V::zeros(),
            k: [V::zeros(); 3],
            y_stage: [V::zeros(); 3],
            f_at_stages: [V::zeros(); 3],
            a: a_t,
            b_higher: b_higher_t,
            b_lower: b_lower_t,
            c: c_t,
            cont: [V::zeros(); 4],
            // Adaptive defaults
            rtol: T::from_f64(1.0e-6).unwrap(),
            atol: T::from_f64(1.0e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10000,
            max_rejects: 100,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            // Implicit defaults
            max_iter: 50,
            tol: T::from_f64(1e-8).unwrap(),
            fd_epsilon_sqrt: T::zero(),
            // Status
            reject: false,
            n_stiff: 0,
            steps: 0,
            status: Status::Uninitialized,
            // Initialize nalgebra structures (empty, to be sized in init)
            jacobian_matrix: nalgebra::DMatrix::zeros(0, 0),
            newton_matrix: nalgebra::DMatrix::zeros(0, 0),
            rhs_newton: nalgebra::DVector::zeros(0),
            delta_k_vec: nalgebra::DVector::zeros(0),
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for Radau5<T, V, D> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Calculate initial derivative f(t0, y0)
        let mut initial_dydt = V::zeros();
        ode.diff(t0, y0, &mut initial_dydt);
        evals.fcn += 1;

        // If h0 is zero calculate h0 using initial derivative
        if self.h0 == T::zero() {
            self.h0 = h_init(ode, t0, tf, y0, 5, self.rtol, self.atol, self.h_min, self.h_max); // Order 5
        }

        // Check bounds
        self.h = validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf)?;
        self.h_prev_step = self.h;

        // Initialize Statistics
        self.reject = false;
        self.n_stiff = 0;
        self.steps = 0;

        // Initialize State
        self.t = t0;
        self.y = *y0;
        self.dydt = initial_dydt; // Store f(t0, y0)

        // Initialize previous state (same as current initially)
        self.t_prev = t0;
        self.y_prev = *y0;
        self.dydt_prev = initial_dydt;

        // Initialize fd_epsilon_sqrt
        self.fd_epsilon_sqrt = T::default_epsilon().sqrt();

        // Initialize Status
        self.status = Status::Initialized;

        // Initialize Jacobian and Newton-related matrices/vectors with correct dimensions
        let dim = y0.len();
        self.jacobian_matrix = nalgebra::DMatrix::zeros(dim, dim);
        let newton_system_size = 3 * dim; // 3 stages for Radau5
        self.newton_matrix = nalgebra::DMatrix::zeros(newton_system_size, newton_system_size);
        self.rhs_newton = nalgebra::DVector::zeros(newton_system_size);
        self.delta_k_vec = nalgebra::DVector::zeros(newton_system_size);
        self.f_at_stages = [V::zeros(); 3]; 

        // Initialize dense output coefficients
        self.cont = [V::zeros(); 4];

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check step size validity
        if self.h.abs() < self.h_min || self.h.abs() < T::default_epsilon() {
            self.status = Status::Error(Error::StepSize { t: self.t, y: self.y });
            return Err(Error::StepSize { t: self.t, y: self.y });
        }

        // Check max steps
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps { t: self.t, y: self.y });
            return Err(Error::MaxSteps { t: self.t, y: self.y });
        }
        self.steps += 1;

        // --- Newton Iteration for stage derivatives k_i ---
        // Initial guess for k_i: k_i^(0) = f(t_n, y_n) (stored in self.dydt)
        for i in 0..3 { 
            self.k[i] = self.dydt;
        }

        // Calculate Jacobian J_n = df/dy(t_n, y_n) once per step attempt
        ode.jacobian(self.t, &self.y, &mut self.jacobian_matrix);
        evals.jac += 1;

        // Form Newton iteration matrix: M = I - h * (A ⊗ J)
        let newton_converged = self.newton_iteration(ode, &mut evals)?;
        
        if !newton_converged {
            self.h *= T::from_f64(0.25).unwrap();
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
            self.reject = true;
            self.n_stiff += 1;
            evals.fcn += 1;

            if self.n_stiff >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness { t: self.t, y: self.y });
                return Err(Error::Stiffness { t: self.t, y: self.y });
            }
            return Ok(evals); // Step rejection
        }

        // --- Newton iteration converged, compute solutions and error ---
        let mut delta_y_high = V::zeros();
        for i in 0..3 { 
            delta_y_high += self.k[i] * (self.b_higher[i] * self.h);
        }
        let y_high = self.y + delta_y_high;

        let mut delta_y_low = V::zeros();
        for i in 0..3 { 
            delta_y_low += self.k[i] * (self.b_lower[i] * self.h);
        }
        let y_low = self.y + delta_y_low;

        let err_vec = y_high - y_low;

        // Calculate scaled error norm
        let mut err_norm = T::zero();
        for n in 0..self.y.len() {
            let scale = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
            if scale > T::zero() {
                err_norm = err_norm.max((err_vec.get(n) / scale).abs());
            }
        }
        err_norm = err_norm.max(T::default_epsilon() * T::from_f64(100.0).unwrap());

        // Calculate new step size based on error
        let order_p1 = T::from_usize(5 + 1).unwrap(); // Order 5 for Radau5
        let mut scale = self.safety_factor * err_norm.powf(-T::one() / order_p1);
        scale = scale.max(self.min_scale).min(self.max_scale);
        let h_new = self.h * scale;

        if err_norm <= T::one() {
            // Step accepted
            self.status = Status::Solving;
            
            // Save current step for interpolation
            self.h_prev_step = self.h;
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.dydt;

            // Update state
            self.t += self.h;
            self.y = y_high;

            // Compute new derivative for next step
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;

            // Calculate dense output coefficients
            self.compute_dense_output_coeffs();

            if self.reject {
                self.n_stiff = 0;
                self.reject = false;
            }
            self.h = constrain_step_size(h_new, self.h_min, self.h_max);
        } else {
            // Step rejected
            self.status = Status::RejectedStep;
            self.reject = true;
            self.n_stiff += 1;

            if self.n_stiff >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness { t: self.t, y: self.y });
                return Err(Error::Stiffness { t: self.t, y: self.y });
            }

            self.h = constrain_step_size(h_new, self.h_min, self.h_max);
            return Ok(evals); // Step rejection
        }

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

impl<T: Real, V: State<T>, D: CallBackData> Radau5<T, V, D> {
    /// Newton iteration for solving the implicit equations for stage derivatives k_i.
    fn newton_iteration<F>(&mut self, ode: &F, evals: &mut Evals) -> Result<bool, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let dim = self.y.len();
        
        // Form Newton matrix M = I_block - h * (A_butcher ⊗ J_f)
        // where I_block is a block identity matrix.
        for i in 0..3 { // block row index
            for l in 0..3 { // block column index
                let scale_factor = -self.h * self.a[i][l];
                for r_idx in 0..dim { // row index within the block
                    for c_idx in 0..dim { // column index within the block
                        self.newton_matrix[(i * dim + r_idx, l * dim + c_idx)] = 
                            self.jacobian_matrix[(r_idx, c_idx)] * scale_factor;
                    }
                }
                if i == l { // If it's a diagonal block, add Identity
                    for d_idx in 0..dim {
                        self.newton_matrix[(i * dim + d_idx, l * dim + d_idx)] += T::one();
                    }
                }
            }
        }

        let mut converged = false;
        
        for _iter in 0..self.max_iter {
            // 1. Compute residual phi(K_current) and store -phi in rhs_newton
            //    y_stage[i] = y_n + h * sum_j(a[i][j] * k[j])
            //    rhs_newton_i = f(t_n + c_i*h, y_stage[i]) - k[i] (this is phi_i)
            for i in 0..3 { 
                self.y_stage[i] = self.y; 
                for j in 0..3 { 
                    self.y_stage[i] += self.k[j] * (self.a[i][j] * self.h);
                }

                ode.diff(self.t + self.c[i] * self.h, &self.y_stage[i], &mut self.f_at_stages[i]);
                evals.fcn += 1;

                for row_idx in 0..dim {
                    self.rhs_newton[i * dim + row_idx] = self.f_at_stages[i].get(row_idx) - self.k[i].get(row_idx);
                }
            }

            // 2. Solve M * delta_k_vec = rhs_newton
            let lu_decomp = nalgebra::LU::new(self.newton_matrix.clone());
            if let Some(solution) = lu_decomp.solve(&self.rhs_newton) {
                self.delta_k_vec.copy_from(&solution);
            } else {
                return Ok(false); // Singular matrix
            }

            // 3. Update K: self.k[i] += delta_k_vec_i
            let mut norm_delta_k_sq = T::zero();
            for i in 0..3 { 
                for row_idx in 0..dim {
                    let delta_val = self.delta_k_vec[i * dim + row_idx];
                    let current_val = self.k[i].get(row_idx);
                    self.k[i].set(row_idx, current_val + delta_val);
                    norm_delta_k_sq += delta_val * delta_val;
                }
            }

            // 4. Check convergence
            let dyno = norm_delta_k_sq.sqrt();
            if dyno < self.tol {
                converged = true;
                break;
            }
        }

        Ok(converged)
    }

    /// Compute the coefficients for dense output after a successful step.
    /// Follows the original RADAU5 Fortran implementation.
    fn compute_dense_output_coeffs(&mut self) {
        // self.cont[0] stores y at the current time t (y_{n+1})
        self.cont[0] = self.y;
        
        // Time points from self.c array:
        // self.c[0] (initialized from C0) corresponds to Fortran C1
        // self.c[1] (initialized from C1) corresponds to Fortran C2
        // self.c[2] (initialized from C2) is 1.0
        let c1_f = self.c[0]; 
        let c2_f = self.c[1]; 

        let c1m1 = c1_f - T::one(); // C1 - 1
        let c2m1 = c2_f - T::one(); // C2 - 1
        let c1mc2 = c1_f - c2_f;   // C1 - C2
        
        // The Fortran code uses Z1I, Z2I, Z3I for dense output coefficient calculation.
        // These Zs are y(x_n + c_i*h) - y_n.
        // In our code, self.y_stage[i] = y_n + h * sum(a_ij * k_j) = Y_i.
        // self.y_prev is y_n at the point this function is called.
        // So, (self.y_stage[i] - self.y_prev) corresponds to Fortran's Z_i values.
        let z1_val = self.y_stage[0] - self.y_prev; 
        let z2_val = self.y_stage[1] - self.y_prev; 
        let z3_val = self.y_stage[2] - self.y_prev; 
        
        // Fortran formulas for CONT coefficients (translated):
        // cont[1] = (Z2-Z3)/C2M1
        // ak = (Z1-Z2)/C1MC2
        // acont3_temp = Z1/C1
        // acont3_temp = (ak-acont3_temp)/C2
        // cont[2] = (ak-cont[1])/C1M1
        // cont[3] = cont[2]-acont3_temp
        
        // Note: self.cont[0] is y_n+1 (current y).
        // self.cont[1] is the coefficient for s in the interpolation polynomial.
        // self.cont[2] is the coefficient for s*(s-c2m1).
        // self.cont[3] is the coefficient for s*(s-c2m1)*(s-c1m1).

        self.cont[1] = (z2_val - z3_val) / c2m1;
        
        let ak = (z1_val - z2_val) / c1mc2;
        
        let mut acont3_temp = z1_val / c1_f;
        
        acont3_temp = (ak - acont3_temp) / c2_f;
        
        self.cont[2] = (ak - self.cont[1]) / c1m1;
        
        self.cont[3] = self.cont[2] - acont3_temp;
    }
}

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for Radau5<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }
        
        // Normalized time: s = (t_interp - t_curr) / h_prev_step
        // This makes s range from -1 (at t_prev) to 0 (at t_curr), matching Fortran's S in CONTR5.
        let s = (t_interp - self.t) / self.h_prev_step;
        
        // Polynomial from CONTR5 function in radau5.f:
        // CONTR5 = CONT(I) + S*(CONT(I+NN)+(S-C2M1)*(CONT(I+NN2)+(S-C1M1)*CONT(I+NN3)))
        // CONT(I) is self.cont[0] (which is y_curr = y_{n+1})
        // S is our s.
        // CONT(I+NN) is self.cont[1], etc.
        // C1M1 and C2M1 are (self.c[0]-1) and (self.c[1]-1) respectively.
        
        let c1_f = self.c[0]; // Corresponds to Fortran C1
        let c2_f = self.c[1]; // Corresponds to Fortran C2
        let c1m1 = c1_f - T::one();
        let c2m1 = c2_f - T::one();
        
        // Interpolation polynomial:
        // y_interp = y_curr + s * (d1 + (s - (c2-1)) * (d2 + (s - (c1-1)) * d3))
        // where d1=self.cont[1], d2=self.cont[2], d3=self.cont[3]
        let y_interp = self.cont[0] + (self.cont[1] + (self.cont[2] + self.cont[3] * (s - c1m1)) * (s - c2m1)) * s;
        
        Ok(y_interp)
    }
}

// Builder pattern methods
impl<
    T: Real,
    V: State<T>,
    D: CallBackData,
> Radau5<T, V, D> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn h0(mut self, h0: T) -> Self { self.h0 = h0; self }
    pub fn rtol(mut self, rtol: T) -> Self { self.rtol = rtol; self }
    pub fn atol(mut self, atol: T) -> Self { self.atol = atol; self }
    pub fn h_min(mut self, h_min: T) -> Self { self.h_min = h_min; self }
    pub fn h_max(mut self, h_max: T) -> Self { self.h_max = h_max; self }
    pub fn max_steps(mut self, max_steps: usize) -> Self { self.max_steps = max_steps; self }
    pub fn max_rejects(mut self, max_rejects: usize) -> Self { self.max_rejects = max_rejects; self }
    pub fn safety_factor(mut self, safety_factor: T) -> Self { self.safety_factor = safety_factor; self }
    pub fn min_scale(mut self, min_scale: T) -> Self { self.min_scale = min_scale; self }
    pub fn max_scale(mut self, max_scale: T) -> Self { self.max_scale = max_scale; self }
    pub fn max_iter(mut self, iter: usize) -> Self { self.max_iter = iter; self }
    pub fn tol(mut self, tol: T) -> Self { self.tol = tol; self }
}