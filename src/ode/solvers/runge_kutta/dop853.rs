//! DOP853 Solver for Ordinary Differential Equations.

use crate::ode::{Solver, SolverStatus, System, EventData};
use crate::traits::Real;
use crate::ode::solvers::utils::{constrain_step_size, validate_step_size_parameters};
use nalgebra::SMatrix;

/// Dormand Prince 8(5, 3) Method for solving ordinary differential equations.
/// 8th order Dormand Prince method with 5th order error estimation and 3rd order interpolation.
/// 
/// Builds should begin with weight, normal, dense, or even methods.
/// and then chain the other methods to set the parameters.
/// The defaults should be great for most cases.
/// 
/// # Example
/// ```
/// use differential_equations::ode::*;
/// use nalgebra::{SVector, vector};
/// 
/// let mut dop853 = DOP853::new()
///    .rtol(1e-12)
///    .atol(1e-12);
/// 
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// struct Example;
/// impl System<f64, 2, 1> for Example {
///    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///       dydt[0] = y[1];
///       dydt[1] = -y[0];
///   }
/// }
/// let solution = IVP::new(Example, t0, tf, y0).solve(&mut dop853).unwrap();
/// 
/// let (t, y) = solution.last().unwrap();
/// println!("Solution: ({}, {})", t, y);
/// ```
/// 
/// # Settings
/// * `rtol`   - Relative tolerance for the solver.
/// * `atol`   - Absolute tolerance for the solver.
/// * `h0`     - Initial step size.
/// * `h_max`   - Maximum step size for the solver.
/// * `max_steps` - Maximum number of steps for the solver.
/// * `n_stiff` - Number of steps to check for stiffness.
/// * `safe`   - Safety factor for step size prediction.
/// * `fac1`   - Parameter for step size selection.
/// * `fac2`   - Parameter for step size selection.
/// * `beta`   - Beta for stabilized step size control.
/// 
/// # Default Settings
/// * `rtol`   - 1e-3
/// * `atol`   - 1e-6
/// * `h0`     - None (Calculated by solver if None)
/// * `h_max`   - None (Calculated by tf - t0 if None)
/// * `h_min`   - 0.0
/// * `max_steps` - 1_000_000
/// * `n_stiff` - 100
/// * `safe`   - 0.9
/// * `fac1`   - 0.33
/// * `fac2`   - 6.0
/// * `beta`   - 0.0
/// 
pub struct DOP853<T: Real, const R: usize, const C: usize, E: EventData> {
    // Initial Conditions
    pub h0: T,                // Initial Step Size

    // Final Time to Solve to
    tf: T,

    // Current iteration
    t: T,
    y: SMatrix<T, R, C>,
    h: T,

    // Tolerances 
    pub rtol: T,
    pub atol: T,

    // Settings
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
    pub n_stiff: usize,

    // DOP853 Specific Settings
    pub safe: T,
    pub fac1: T,
    pub fac2: T,
    pub beta: T,

    // Derived Settings
    expo1: T,
    facc1: T,
    facc2: T,
    facold: T,
    fac11: T,
    fac: T,

    // Iteration Tracking
    status: SolverStatus<T, R, C, E>,

    // Stiffness Detection
    h_lamb: T,
    non_stiff_counter: usize,
    stiffness_counter: usize,

    // Butcher tableau coefficients (converted to type T)
    a: [[T; 12]; 12],
    b: [T; 12],
    c: [T; 12],
    er: [T; 12],
    bhh: [T; 3],
    
    // Dense output coefficients
    a_dense: [[T; 16]; 3],
    c_dense: [T; 3],
    dense: [[T; 16]; 4],

    // Statistics
    evals: usize,
    steps: usize,
    rejected_steps: usize,
    accepted_steps: usize,

    // Derivatives - using array instead of individually numbered variables
    k: [SMatrix<T, R, C>; 12],  // k[0] is derivative at t, others are stage derivatives

    // For Interpolation - using array instead of individually numbered variables
    cached_step_num: usize,
    y_old: SMatrix<T, R, C>, // State at Previous Step
    k_old: SMatrix<T, R, C>, // Derivative at Previous Step
    t_old: T, // Time of Previous Step
    h_old: T, // Step Size of Previous Step
    cont: [SMatrix<T, R, C>; 8], // Interpolation coefficients
}

impl<T: Real, const R: usize, const C: usize, E: EventData> Solver<T, R, C, E> for DOP853<T, R, C, E> {    
    fn init<S>(&mut self, system: &S, t0: T, tf: T, y0: &SMatrix<T, R, C>)  -> Result<(), SolverStatus<T, R, C, E>>
    where 
        S: System<T, R, C, E>
    {
        // Set tf so step size doesn't go past it
        self.tf = tf;

        // Initialize Statistics
        self.evals = 0;
        self.steps = 0;
        self.rejected_steps = 0;
        self.accepted_steps = 0;

        // Set Current State as Initial State
        self.t = t0;
        self.y = y0.clone();

        // Calculate derivative at t0
        system.diff(t0, y0, &mut self.k[0]);
        self.evals += 1;

        // Initialize Previous State
        self.t_old = self.t;
        self.y_old = self.y;
        self.k_old = self.k[0];

        // Calculate Initial Step
        if self.h0 == T::zero() {
            self.h_init(system, t0, tf);

            // Adjust h0 to be within bounds
            let posneg = (tf - t0).signum();
            if self.h0.abs() < self.h_min.abs() {
                self.h0 = self.h_min.abs() * posneg;
            } else if self.h0.abs() > self.h_max.abs() {
                self.h0 = self.h_max.abs() * posneg;
            }
        }

        // Check if h0 is within bounds, and h_min and h_max are valid
        match validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Set h_max to prevent single step overshoot
        self.h_max = match self.h_max {
            x if x > (self.tf - t0).abs() => (self.tf - t0).abs(),
            _ => self.h_max.abs(),
        };

        // Make sure iteration variables are reset
        self.h_lamb = T::zero();
        self.non_stiff_counter = 0;
        self.stiffness_counter = 0;

        // Solver is ready to go
        self.status = SolverStatus::Initialized;

        Ok(())
    }

    fn step<S>(&mut self, system: &S) 
    where 
        S: System<T, R, C, E>
    {
        // Check if Max Steps Reached
        if self.steps >= self.max_steps {
            self.status = SolverStatus::MaxSteps(self.t, self.y.clone());
            return;
        }
    
        // Check if Step Size is too smaller then machine default_epsilon
        if self.h.abs() < T::default_epsilon() {
            self.status = SolverStatus::StepSize(self.t, self.y.clone());
            return;
        }
    
        // The twelve stages
        system.diff(
            self.t + self.c[1] * self.h,
            &(self.y + self.k[0] * (self.a[1][0] * self.h)),
            &mut self.k[1]
        );
        system.diff(
            self.t + self.c[2] * self.h,
            &(self.y + self.k[0] * (self.a[2][0] * self.h) + self.k[1] * (self.a[2][1] * self.h)),
            &mut self.k[2]
        );
        system.diff(
            self.t + self.c[3] * self.h,
            &(self.y + self.k[0] * (self.a[3][0] * self.h) + self.k[2] * (self.a[3][2] * self.h)),
            &mut self.k[3]
        );
        system.diff(
            self.t + self.c[4] * self.h,
            &(self.y + self.k[0] * (self.a[4][0] * self.h) + self.k[2] * (self.a[4][2] * self.h) + self.k[3] * (self.a[4][3] * self.h)),
            &mut self.k[4]
        );
        system.diff(
            self.t + self.c[5] * self.h,
            &(self.y + self.k[0] * (self.a[5][0] * self.h) + self.k[3] * (self.a[5][3] * self.h) + self.k[4] * (self.a[5][4] * self.h)),
            &mut self.k[5]
        );
        system.diff(
            self.t + self.c[6] * self.h,
            &(self.y + self.k[0] * (self.a[6][0] * self.h) + self.k[3] * (self.a[6][3] * self.h) + self.k[4] * (self.a[6][4] * self.h) + self.k[5] * (self.a[6][5] * self.h)),
            &mut self.k[6]
        );
        system.diff(
            self.t + self.c[7] * self.h,
            &(self.y + self.k[0] * (self.a[7][0] * self.h) + self.k[3] * (self.a[7][3] * self.h) + self.k[4] * (self.a[7][4] * self.h) + self.k[5] * (self.a[7][5] * self.h) + self.k[6] * (self.a[7][6] * self.h)),
            &mut self.k[7]
        );
        system.diff(
            self.t + self.c[8] * self.h,
            &(self.y + self.k[0] * (self.a[8][0] * self.h) + self.k[3] * (self.a[8][3] * self.h) + self.k[4] * (self.a[8][4] * self.h) + self.k[5] * (self.a[8][5] * self.h) + self.k[6] * (self.a[8][6] * self.h) + self.k[7] * (self.a[8][7] * self.h)),
            &mut self.k[8]
        );
        system.diff(
            self.t + self.c[9] * self.h,
            &(self.y + self.k[0] * (self.a[9][0] * self.h) + self.k[3] * (self.a[9][3] * self.h) + self.k[4] * (self.a[9][4] * self.h) + self.k[5] * (self.a[9][5] * self.h) + self.k[6] * (self.a[9][6] * self.h) + self.k[7] * (self.a[9][7] * self.h) + self.k[8] * (self.a[9][8] * self.h)),
            &mut self.k[9]
        );
        system.diff(
            self.t + self.c[10] * self.h,
            &(self.y + self.k[0] * (self.a[10][0] * self.h) + self.k[3] * (self.a[10][3] * self.h) + self.k[4] * (self.a[10][4] * self.h) + self.k[5] * (self.a[10][5] * self.h) + self.k[6] * (self.a[10][6] * self.h) + self.k[7] * (self.a[10][7] * self.h) + self.k[8] * (self.a[10][8] * self.h) + self.k[9] * (self.a[10][9] * self.h)),
            &mut self.k[1]
        );
        let t_new = self.t + self.h;
        let yy1 = self.y + self.k[0] * (self.a[11][0] * self.h) + self.k[3] * (self.a[11][3] * self.h) + self.k[4] * (self.a[11][4] * self.h) + self.k[5] * (self.a[11][5] * self.h) + self.k[6] * (self.a[11][6] * self.h) + self.k[7] * (self.a[11][7] * self.h) + self.k[8] * (self.a[11][8] * self.h) + self.k[9] * (self.a[11][9] * self.h) + self.k[1] * (self.a[11][10] * self.h);
        system.diff(
            t_new,
            &yy1,
            &mut self.k[2]
        );
        self.k[3] = self.k[0] * self.b[0] + self.k[5] * self.b[5] + self.k[6] * self.b[6] + self.k[7] * self.b[7] + 
                  self.k[8] * self.b[8] + self.k[9] * self.b[9] + self.k[1] * self.b[10] + self.k[2] * self.b[11];
        self.k[4] = self.y + self.k[3] * self.h;
        
        self.evals += 11;
    
        // Error Estimation
        let mut err = T::zero();
        let mut err2 = T::zero();
    
        let n = self.y.len();
        for i in 0..n {
            let sk = self.atol + self.rtol * self.y[i].abs().max(self.k[4][i].abs());
            let erri = self.k[3][i] - self.bhh[0] * self.k[0][i] - self.bhh[1] * self.k[8][i] - self.bhh[2] * self.k[2][i];
            err2 = err2 + (erri / sk).powi(2);
            let erri = self.er[0] * self.k[0][i]
                     + self.er[5] * self.k[5][i]
                     + self.er[6] * self.k[6][i]
                     + self.er[7] * self.k[7][i]
                     + self.er[8] * self.k[8][i]
                     + self.er[9] * self.k[9][i]
                     + self.er[10] * self.k[1][i]
                     + self.er[11] * self.k[2][i];
            err = err + (erri / sk).powi(2);
        }
        let mut deno = err + T::from_f64(0.01).unwrap() * err2;
        if deno <= T::zero() {
            deno = T::one();
        }
        err = self.h.abs() * err * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();
    
        // Computation of h_new
        self.fac11 = err.powf(self.expo1);
        // Lund-stabilization
        self.fac = self.fac11 / self.facold.powf(self.beta);
        // Requirement that fac1 <= h_new/h <= fac2
        self.fac = self.facc2.max(self.facc1.min(self.fac / self.safe));
        let mut h_new = self.h / self.fac;
    
        if err <= T::one() {
            // Step Accepted
            self.facold = err.max(T::from_f64(1.0e-4).unwrap());
            self.accepted_steps += 1;
            let y_new = self.k[4];
            system.diff(t_new, &y_new, &mut self.k[3]);
            self.evals += 1;
    
            // stiffness detection
            if self.accepted_steps % self.n_stiff == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = self.k[3] - self.k[2];
                stdnum += sqr.component_mul(&sqr).sum();
                let sqr = self.k[4] - yy1;
                stden += sqr.component_mul(&sqr).sum();
    
                if stden > T::zero() {
                    self.h_lamb = self.h * (stdnum / stden).sqrt();
                }
                if self.h_lamb > T::from_f64(6.1).unwrap() {
                    self.non_stiff_counter = 0;
                    self.stiffness_counter += 1;
                    if self.stiffness_counter == 15 {
                        // Early Exit Stiffness Detected
                        self.status = SolverStatus::Stiffness(self.t, self.y.clone());
                        return;
                    }
                } else {
                    self.non_stiff_counter += 1;
                    if self.non_stiff_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }
    
            // For Interpolation
            self.y_old = self.y;
            self.k_old = self.k[0];
            self.t_old = self.t;
            self.h_old = self.h;
    
            // Update State
            self.k[0] = self.k[3];
            self.y = self.k[4];
            self.t = t_new;
    
            // Check if previous step rejected
            if let SolverStatus::RejectedStep = self.status {
                h_new = self.h.min(h_new);
                self.status = SolverStatus::Solving;
            }
        } else {
            // Step Rejected
            h_new = self.h / self.facc1.min(self.fac11 / self.safe);
            self.status = SolverStatus::RejectedStep;
            self.rejected_steps += 1;
        }
        // Step Complete
        self.h = constrain_step_size(h_new, self.h_min, self.h_max);
    
        self.steps += 1;
    }

    fn interpolate<F>(&mut self, system: &F, t: T) -> SMatrix<T, R, C>
    where
        F: System<T, R, C, E>
    {
        if self.cached_step_num != self.steps {
            // Initial coefficients
            self.cont[0] = self.y_old;
            let ydiff = self.k[4] - self.y_old;
            self.cont[1] = ydiff;
            let bspl = self.k_old * self.h_old - ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.k[3] * self.h_old - bspl;
            
            // Use stored dense output coefficients instead of direct T::from_f64() calls
            self.cont[4] = self.k_old * self.dense[0][0] + 
                           self.k[5] * self.dense[0][5] + 
                           self.k[6] * self.dense[0][6] + 
                           self.k[7] * self.dense[0][7] +
                           self.k[8] * self.dense[0][8] + 
                           self.k[9] * self.dense[0][9] + 
                           self.k[1] * self.dense[0][10] + 
                           self.k[2] * self.dense[0][11];
                         
            self.cont[5] = self.k_old * self.dense[1][0] + 
                           self.k[5] * self.dense[1][5] + 
                           self.k[6] * self.dense[1][6] + 
                           self.k[7] * self.dense[1][7] +
                           self.k[8] * self.dense[1][8] + 
                           self.k[9] * self.dense[1][9] + 
                           self.k[1] * self.dense[1][10] + 
                           self.k[2] * self.dense[1][11];
                         
            self.cont[6] = self.k_old * self.dense[2][0] + 
                           self.k[5] * self.dense[2][5] + 
                           self.k[6] * self.dense[2][6] + 
                           self.k[7] * self.dense[2][7] +
                           self.k[8] * self.dense[2][8] + 
                           self.k[9] * self.dense[2][9] + 
                           self.k[1] * self.dense[2][10] + 
                           self.k[2] * self.dense[2][11];
                         
            self.cont[7] = self.k_old * self.dense[3][0] + 
                           self.k[5] * self.dense[3][5] + 
                           self.k[6] * self.dense[3][6] + 
                           self.k[7] * self.dense[3][7] +
                           self.k[8] * self.dense[3][8] + 
                           self.k[9] * self.dense[3][9] + 
                           self.k[1] * self.dense[3][10] + 
                           self.k[2] * self.dense[3][11];
    
            // Next 3 Function Evaluations - using the stored a_dense and c_dense arrays
            system.diff(
                self.t_old + self.c_dense[0] * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a_dense[0][0] + 
                    self.k[6] * self.a_dense[0][6] + 
                    self.k[7] * self.a_dense[0][7] + 
                    self.k[8] * self.a_dense[0][8] + 
                    self.k[9] * self.a_dense[0][9] + 
                    self.k[1] * self.a_dense[0][10] + 
                    self.k[2] * self.a_dense[0][11] +
                    self.k[3] * self.a_dense[0][12]) * self.h_old
                ),
                &mut self.k[9]
            );
            
            system.diff(
                self.t_old + self.c_dense[1] * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a_dense[1][0] + 
                    self.k[5] * self.a_dense[1][5] + 
                    self.k[6] * self.a_dense[1][6] + 
                    self.k[7] * self.a_dense[1][7] + 
                    self.k[1] * self.a_dense[1][10] + 
                    self.k[2] * self.a_dense[1][11] + 
                    self.k[3] * self.a_dense[1][12] + 
                    self.k[9] * self.a_dense[1][13]) * self.h_old
                ),
                &mut self.k[1]
            );
            
            system.diff(
                self.t_old + self.c_dense[2] * self.h_old,
                &(self.y_old + (
                    self.k_old * self.a_dense[2][0] + 
                    self.k[5] * self.a_dense[2][5] + 
                    self.k[6] * self.a_dense[2][6] + 
                    self.k[7] * self.a_dense[2][7] + 
                    self.k[8] * self.a_dense[2][8] + 
                    self.k[3] * self.a_dense[2][12] + 
                    self.k[9] * self.a_dense[2][13] + 
                    self.k[1] * self.a_dense[2][14]) * self.h_old
                ),
                &mut self.k[2]
            );
            self.evals += 3;
    
            // Final preparation - add contributions from the extra stages and scale
            self.cont[4] = (self.cont[4] + 
                           self.k[3] * self.dense[0][12] + 
                           self.k[9] * self.dense[0][13] + 
                           self.k[1] * self.dense[0][14] + 
                           self.k[2] * self.dense[0][15]) * self.h_old;
                         
            self.cont[5] = (self.cont[5] + 
                           self.k[3] * self.dense[1][12] + 
                           self.k[9] * self.dense[1][13] + 
                           self.k[1] * self.dense[1][14] + 
                           self.k[2] * self.dense[1][15]) * self.h_old;
                         
            self.cont[6] = (self.cont[6] + 
                           self.k[3] * self.dense[2][12] + 
                           self.k[9] * self.dense[2][13] + 
                           self.k[1] * self.dense[2][14] + 
                           self.k[2] * self.dense[2][15]) * self.h_old;
                         
            self.cont[7] = (self.cont[7] + 
                           self.k[3] * self.dense[3][12] + 
                           self.k[9] * self.dense[3][13] + 
                           self.k[1] * self.dense[3][14] + 
                           self.k[2] * self.dense[3][15]) * self.h_old;
    
            // Step is cached
            self.cached_step_num = self.steps;
        }
    
        // Evaluate the interpolation polynomial at the requested time
        let s = (t - self.t_old) / self.h_old;
        let s1 = T::one() - s;
    
        // Compute the interpolated value using nested polynomial evaluation
        let conpar = self.cont[4] + (self.cont[5] + (self.cont[6] + self.cont[7] * s) * s1) * s;
        let y = self.cont[0] + (self.cont[1] + (self.cont[2] + (self.cont[3] + conpar * s1) * s) * s1) * s;
    
        y
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &SMatrix<T, R, C> {
        &self.y
    }

    fn dydt(&self) -> &SMatrix<T, R, C> {
        &self.k[0]
    }

    fn t_prev(&self) -> T {
        self.t_old
    }

    fn y_prev(&self) -> &SMatrix<T, R, C> {
        &self.y_old
    }

    fn dydt_prev(&self) -> &SMatrix<T, R, C> {
        &self.k_old
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        self.h = h;
    }

    fn evals(&self) -> usize {
        self.evals
    }

    fn steps(&self) -> usize {
        self.steps
    }

    fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }

    fn accepted_steps(&self) -> usize {
        self.steps
    }

    fn status(&self) -> &SolverStatus<T, R, C, E> {
        &self.status
    }

    fn set_status(&mut self, status: SolverStatus<T, R, C, E>) {
        self.status = status;
    }
}

impl<T: Real, const R: usize, const C: usize, E: EventData> DOP853<T, R, C, E> {
    /// Creates a new DOP853 Solver.
    /// 
    /// # Returns
    /// * `system` - Function that defines the ordinary differential equation dy/dt = f(t, y).
    /// # Returns
    /// * DOP853 Struct ready to go for solving.
    ///  
    pub fn new() -> Self {
        DOP853 {
            ..Default::default()
        }
    }

    /// Initializes the initial step size for the solver.
    /// The initial step size is computed such that h**8 * f0.norm().max(der2.norm()) = 0.01
    /// 
    /// This function is called internally by the init function if non initial step size, h, is not provided.
    /// This function also dependents on derived settings and the initial derivative vector.
    /// Thus it is private and should not be called directly by users.
    /// 
    /// # Arguments
    /// * `ode` - Function that defines the ordinary differential equation dy/dt = f(t, y).
    /// 
    /// # Returns
    /// * Updates self.h with the initial step size.
    /// 
    fn h_init<S>(&mut self, system: &S, t0: T, tf: T)
    where 
        S: System<T, R, C, E>
    {
        // Set the initial step size h0 to h, if its 0.0 then it will be calculated
        self.h = self.h0;

        let posneg = (tf - t0).signum();

        let sk = (self.y.abs() * self.rtol).add_scalar(self.atol);
        let sqr = self.k[0].component_div(&sk);
        let dnf = sqr.component_mul(&sqr).sum();
        let sqr = self.y.component_div(&sk);
        let dny = sqr.component_mul(&sqr).sum();

        self.h = if (dnf <= T::from_f64(1.0e-10).unwrap()) || (dny <= T::from_f64(1.0e-10).unwrap()) {
            T::from_f64(1.0e-6).unwrap()
        } else {
            (dny / dnf).sqrt() * T::from_f64(0.01).unwrap()
        };

        self.h = self.h.min(self.h_max);
        self.h = if posneg < T::zero() {
            -self.h.abs()
        } else {
            self.h.abs()
        };

        // perform an explicit Euler step
        system.diff(self.t + self.h, &(self.y + (self.k[0] * self.h)), &mut self.k[1]);
        self.evals += 1;

        // estimate the second derivative of the solution
        let sk = (self.y.abs() * self.rtol).add_scalar(self.atol);
        let sqr = (self.k[1] - self.k[0]).component_div(&sk);
        let der2 = (sqr.component_mul(&sqr)).sum().sqrt() / self.h;

        // step size is computed such that h**8 * f0.norm().max(der2.norm()) = 0.01
        let der12 = der2.abs().max(dnf.sqrt());
        let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
            (self.h.abs() * T::from_f64(1.0e-3).unwrap()).max(T::from_f64(1.0e-6).unwrap())
        } else {
            (T::from_f64(0.01).unwrap() / der12).powf(T::one() / T::from_f64(8.0).unwrap())
        };

        self.h = (T::from_f64(100.0).unwrap() * posneg * self.h).min(h1.min(self.h_max));

        // Make sure step is going in the right direction
        self.h = self.h.abs() * posneg;
        self.h0 = self.h;
    }

    // Builder Functions
    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }

    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn n_stiff(mut self, n_stiff: usize) -> Self {
        self.n_stiff = n_stiff;
        self
    }

    pub fn safe(mut self, safe: T) -> Self {
        self.safe = safe;
        self
    }

    pub fn beta(mut self, beta: T) -> Self {
        self.beta = beta;
        self
    }

    pub fn fac1(mut self, fac1: T) -> Self {
        self.fac1 = fac1;
        self
    }

    pub fn fac2(mut self, fac2: T) -> Self {
        self.fac2 = fac2;
        self
    }

    pub fn expo1(mut self, expo1: T) -> Self {
        self.expo1 = expo1;
        self
    }

    pub fn facc1(mut self, facc1: T) -> Self {
        self.facc1 = facc1;
        self
    }

    pub fn facc2(mut self, facc2: T) -> Self {
        self.facc2 = facc2;
        self
    }
}

impl<T: Real, const R: usize, const C: usize, E: EventData> Default for DOP853<T, R, C, E> {
    fn default() -> Self {
        // Convert coefficient arrays from f64 to type T
        let a = DOP853_A.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = DOP853_B.map(|x| T::from_f64(x).unwrap());
        let c = DOP853_C.map(|x| T::from_f64(x).unwrap());
        let er = DOP853_ER.map(|x| T::from_f64(x).unwrap());
        let bhh = DOP853_BHH.map(|x| T::from_f64(x).unwrap());
        
        let a_dense = DOP853_A_DENSE.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let c_dense = DOP853_C_DENSE.map(|x| T::from_f64(x).unwrap());
        let dense = DOP853_DENSE.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        
        // Create arrays of zeros for k and cont matrices
        let k_zeros = [SMatrix::zeros(); 12];
        let cont_zeros = [SMatrix::zeros(); 8];
        
        DOP853 {
            // State Variables
            t: T::zero(),
            y: SMatrix::zeros(),
            h: T::zero(),

            // Settings
            tf: T::zero(),
            h0: T::zero(),
            rtol: T::from_f64(1e-3).unwrap(),
            atol: T::from_f64(1e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 1_000_000,
            n_stiff: 100,
            safe: T::from_f64(0.9).unwrap(),
            fac1: T::from_f64(0.33).unwrap(),
            fac2: T::from_f64(6.0).unwrap(),
            beta: T::from_f64(0.0).unwrap(),
            expo1: T::from_f64(1.0 / 8.0).unwrap(),
            facc1: T::from_f64(1.0 / 0.33).unwrap(),
            facc2: T::from_f64(1.0 / 6.0).unwrap(),
            facold: T::from_f64(1.0e-4).unwrap(),
            fac11: T::zero(),
            fac: T::zero(),
            
            // Butcher Tableau Coefficients
            a,
            b, 
            c,
            er,
            bhh,
            a_dense,
            c_dense,
            dense,
            
            // Status and Counters
            status: SolverStatus::Uninitialized,
            h_lamb: T::zero(),
            non_stiff_counter: 0,
            stiffness_counter: 0,
            evals: 0,
            steps: 0,
            rejected_steps: 0,
            accepted_steps: 0,
            
            // Coefficents and temporary storage
            k: k_zeros,
            cached_step_num: 0,
            y_old: SMatrix::zeros(),
            k_old: SMatrix::zeros(),
            t_old: T::zero(),
            h_old: T::zero(),
            cont: cont_zeros,
        }
    }
}


// DOP853 Butcher Tableau

// 12 Stage Core

// A matrix (12x12, lower triangular)
const DOP853_A: [[f64; 12]; 12] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.26001519587677318785587544488E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.97250569845378994544595329183E-2, 5.91751709536136983633785987549E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.95875854768068491816892993775E-2, 0.0, 8.87627564304205475450678981324E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.41365134159266685502369798665E-1, 0.0, -8.84549479328286085344864962717E-1, 9.24834003261792003115737966543E-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.7037037037037037037037037037E-2, 0.0, 0.0, 1.70828608729473871279604482173E-1, 1.25467687566822425016691814123E-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.7109375E-2, 0.0, 0.0, 1.70252211019544039314978060272E-1, 6.02165389804559606850219397283E-2, -1.7578125E-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.70920001185047927108779319836E-2, 0.0, 0.0, 1.70383925712239993810214054705E-1, 1.07262030446373284651809199168E-1, -1.53194377486244017527936158236E-2, 8.27378916381402288758473766002E-3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.24110958716075717114429577812E-1, 0.0, 0.0, -3.36089262944694129406857109825E0, -8.68219346841726006818189891453E-1, 2.75920996994467083049415600797E1, 2.01540675504778934086186788979E1, -4.34898841810699588477366255144E1, 0.0, 0.0, 0.0, 0.0],
    [4.77662536438264365890433908527E-1, 0.0, 0.0, -2.48811461997166764192642586468E0, -5.90290826836842996371446475743E-1, 2.12300514481811942347288949897E1, 1.52792336328824235832596922938E1, -3.32882109689848629194453265587E1, -2.03312017085086261358222928593E-2, 0.0, 0.0, 0.0],
    [-9.3714243008598732571704021658E-1, 0.0, 0.0, 5.18637242884406370830023853209E0, 1.09143734899672957818500254654E0, -8.14978701074692612513997267357E0, -1.85200656599969598641566180701E1, 2.27394870993505042818970056734E1, 2.49360555267965238987089396762E0, -3.0467644718982195003823669022E0, 0.0, 0.0],
    [2.27331014751653820792359768449E0, 0.0, 0.0, -1.05344954667372501984066689879E1, -2.00087205822486249909675718444E0, -1.79589318631187989172765950534E1, 2.79488845294199600508499808837E1, -2.85899827713502369474065508674E0, -8.87285693353062954433549289258E0, 1.23605671757943030647266201528E1, 6.43392746015763530355970484046E-1, 0.0]
];

// C coefficients (nodes)
const DOP853_C: [f64; 12] = [
    0.0, // C1 (not given in constants, but must be 0)
    0.526001519587677318785587544488E-01, // C2
    0.789002279381515978178381316732E-01, // C3
    0.118350341907227396726757197510E+00, // C4
    0.281649658092772603273242802490E+00, // C5
    0.333333333333333333333333333333E+00, // C6
    0.25E+00,                             // C7
    0.307692307692307692307692307692E+00, // C8
    0.651282051282051282051282051282E+00, // C9
    0.6E+00,                              // C10
    0.857142857142857142857142857142E+00, // C11
    1.0,                                  // C12 (final point, not explicitly given)
];

// B coefficients (weights for main method)
const DOP853_B: [f64; 12] = [
    5.42937341165687622380535766363E-2, // B1
    0.0,                                // B2-B5 are zero (not explicitly listed)
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566E0,  // B6
    1.89151789931450038304281599044E0,  // B7
    -5.8012039600105847814672114227E0,  // B8
    3.1116436695781989440891606237E-1,  // B9
    -1.52160949662516078556178806805E-1, // B10
    2.01365400804030348374776537501E-1, // B11
    4.47106157277725905176885569043E-2, // B12
];

// Error estimation coefficients

// Error estimation coefficients (constructed from ER values)
const DOP853_ER: [f64; 12] = [
    0.1312004499419488073250102996E-01,  // ER1
    0.0,                                 // ER2-ER5 are zero
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753E+01, // ER6
    -0.4957589496572501915214079952E+00, // ER7
    0.1664377182454986536961530415E+01,  // ER8
    -0.3503288487499736816886487290E+00, // ER9
    0.3341791187130174790297318841E+00,  // ER10
    0.8192320648511571246570742613E-01,  // ER11
    -0.2235530786388629525884427845E-01, // ER12
];

const DOP853_BHH: [f64; 3] = [
    0.244094488188976377952755905512E+00, // BHH1
    0.733846688281611857341361741547E+00, // BHH2
    0.220588235294117647058823529412E-01, // BHH3
];

// Dense output Coefficients


// Dense output A coefficients (for the 3 extra stages used in interpolation)
const DOP853_A_DENSE: [[f64; 16]; 3] = [
    // Stage 14 coefficients (C14 = 0.1)
    [
        5.61675022830479523392909219681E-2,    // A141
        0.0, 0.0, 0.0, 0.0, 0.0,               // A142-A146 (zero)
        2.53500210216624811088794765333E-1,    // A147
        -2.46239037470802489917441475441E-1,   // A148
        -1.24191423263816360469010140626E-1,   // A149
        1.5329179827876569731206322685E-1,     // A1410
        8.20105229563468988491666602057E-3,    // A1411
        7.56789766054569976138603589584E-3,    // A1412
        -8.298E-3,                             // A1413
        0.0, 0.0, 0.0                          // A1414-A1416 (zero/not used)
    ],
    // Stage 15 coefficients (C15 = 0.2)
    [
        3.18346481635021405060768473261E-2,    // A151
        0.0, 0.0, 0.0, 0.0,                    // A152-A155 (zero)
        2.83009096723667755288322961402E-2,    // A156
        5.35419883074385676223797384372E-2,    // A157
        -5.49237485713909884646569340306E-2,   // A158
        0.0, 0.0,                              // A159-A1510 (zero)
        -1.08347328697249322858509316994E-4,   // A1511
        3.82571090835658412954920192323E-4,    // A1512
        -3.40465008687404560802977114492E-4,   // A1513
        1.41312443674632500278074618366E-1,    // A1514
        0.0, 0.0                               // A1515-A1516 (zero/not used)
    ],
    // Stage 16 coefficients (C16 = 0.777...)
    [
        -4.28896301583791923408573538692E-1,   // A161
        0.0, 0.0, 0.0, 0.0,                    // A162-A165 (zero)
        -4.69762141536116384314449447206E0,    // A166
        7.68342119606259904184240953878E0,     // A167
        4.06898981839711007970213554331E0,     // A168
        3.56727187455281109270669543021E-1,    // A169
        0.0, 0.0, 0.0,                         // A1610-A1612 (zero)
        -1.39902416515901462129418009734E-3,   // A1613
        2.9475147891527723389556272149E0,      // A1614
        -9.15095847217987001081870187138E0,    // A1615
        0.0                                    // A1616 (not used)
    ]
];

const DOP853_C_DENSE: [f64; 3] = [
    0.1E+00, // C14
    0.2E+00, // C15
    0.777777777777777777777777777778E+00, // C16
];

// Dense output coefficients for stage 4
const DOP853_D4: [f64; 16] = [
    -0.84289382761090128651353491142E+01, // D41
    0.0, 0.0, 0.0, 0.0,                   // D42-D45 are zero
    0.56671495351937776962531783590E+00,  // D46
    -0.30689499459498916912797304727E+01, // D47
    0.23846676565120698287728149680E+01,  // D48
    0.21170345824450282767155149946E+01,  // D49
    -0.87139158377797299206789907490E+00, // D410
    0.22404374302607882758541771650E+01,  // D411
    0.63157877876946881815570249290E+00,  // D412
    -0.88990336451333310820698117400E-01, // D413
    0.18148505520854727256656404962E+02,  // D414
    -0.91946323924783554000451984436E+01, // D415
    -0.44360363875948939664310572000E+01, // D416
];

// Dense output coefficients for stages 5, 6, and 7 follow same pattern
const DOP853_D5: [f64; 16] = [
    0.10427508642579134603413151009E+02, // D51
    0.0, 0.0, 0.0, 0.0,                  // D52-D55 are zero 
    0.24228349177525818288430175319E+03, // D56
    0.16520045171727028198505394887E+03, // D57
    -0.37454675472269020279518312152E+03, // D58
    -0.22113666853125306036270938578E+02, // D59
    0.77334326684722638389603898808E+01, // D510
    -0.30674084731089398182061213626E+02, // D511
    -0.93321305264302278729567221706E+01, // D512
    0.15697238121770843886131091075E+02, // D513
    -0.31139403219565177677282850411E+02, // D514
    -0.93529243588444783865713862664E+01, // D515
    0.35816841486394083752465898540E+02, // D516
];

const DOP853_D6: [f64; 16] = [
    0.19985053242002433820987653617E+02, // D61
    0.0, 0.0, 0.0, 0.0,                  // D62-D65 are zero
    -0.38703730874935176555105901742E+03, // D66
    -0.18917813819516756882830838328E+03, // D67
    0.52780815920542364900561016686E+03,  // D68
    -0.11573902539959630126141871134E+02, // D69
    0.68812326946963000169666922661E+01,  // D610
    -0.10006050966910838403183860980E+01, // D611
    0.77771377980534432092869265740E+00,  // D612
    -0.27782057523535084065932004339E+01, // D613
    -0.60196695231264120758267380846E+02, // D614
    0.84320405506677161018159903784E+02,  // D615
    0.11992291136182789328035130030E+02,  // D616
];

const DOP853_D7: [f64; 16] = [
    -0.25693933462703749003312586129E+02, // D71
    0.0, 0.0, 0.0, 0.0,                  // D72-D75 are zero
    -0.15418974869023643374053993627E+03, // D76
    -0.23152937917604549567536039109E+03, // D77
    0.35763911791061412378285349910E+03,  // D78
    0.93405324183624310003907691704E+02,  // D79
    -0.37458323136451633156875139351E+02, // D710
    0.10409964950896230045147246184E+03,  // D711
    0.29840293426660503123344363579E+02,  // D712
    -0.43533456590011143754432175058E+02, // D713
    0.96324553959188282948394950600E+02,  // D714
    -0.39177261675615439165231486172E+02, // D715
    -0.14972683625798562581422125276E+03, // D716
];

// Dense output coefficients as a 3D array [stage][coefficient_index]
const DOP853_DENSE: [[f64; 16]; 4] = [
    DOP853_D4,
    DOP853_D5,
    DOP853_D6,
    DOP853_D7,
];
