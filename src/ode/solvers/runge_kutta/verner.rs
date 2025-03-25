/// Macro to create a Runge-Kutta solver with dense output capabilities
///
/// # Arguments
/// 
/// * `name`: Name of the solver struct to create
/// * `a`: Matrix of coefficients for intermediate stages
/// * `b`: 2D array where first row is higher order weights, second row is lower order weights
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the method
/// * `stages`: Number of stages in the method
/// * `dense_stages`: Number of terms in the interpolation polynomial
/// * `extra_stages`: Number of additional stages for interpolation
/// * `a_dense`: Coefficients for additional stages needed for interpolation
/// * `c_dense`: Time offsets for additional interpolation stages
/// * `b_dense`: Coefficients for interpolation polynomial
///
/// # Note
/// 
/// This macro generates a full solver with the ability to interpolate the solution
/// at any point within a step. The interpolation capability requires additional
/// function evaluations but provides high-order continuous output.
/// 
#[macro_export]
macro_rules! adaptive_dense_runge_kutta_method {
    (
        $(#[$attr:meta])*
        name: $name:ident,
        a: $a:expr,
        b: $b:expr,
        c: $c:expr,
        order: $order:expr,
        stages: $stages:expr,
        // Interpolation info
        dense_stages: $dense_stages:expr,
        extra_stages: $extra_stages:expr,
        a_dense: $a_dense:expr,
        c_dense: $c_dense:expr,
        b_dense: $b_dense:expr
        $(,)? // Optional trailing comma
    ) => {
        $(#[$attr])*
        #[doc = "\n\n"]
        #[doc = "This adaptive solver with dense output was automatically generated using the `adaptive_dense_runge_kutta_method` macro."]
        pub struct $name<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::traits::EventData> {
            // Initial Step Size
            pub h0: T,

            // Current Step Size
            h: T,

            // Current State
            t: T,
            y: $crate::ode::SMatrix<T, R, C>,
            dydt: $crate::ode::SMatrix<T, R, C>,

            // Previous State
            t_prev: T,
            y_prev: $crate::ode::SMatrix<T, R, C>,
            dydt_prev: $crate::ode::SMatrix<T, R, C>,

            // Stage values (fixed size array of matrices)
            k: [$crate::ode::SMatrix<T, R, C>; $stages + $extra_stages], // Main stages + extra stages for interpolation

            // Constants from Butcher tableau
            a: [[T; $stages]; $stages],
            b_higher: [T; $stages],
            b_lower: [T; $stages],
            c: [T; $stages],
            
            // Interpolation coefficients
            a_dense: [[T; $stages]; $extra_stages],  // Type inferred from a_dense
            c_dense: [T; $extra_stages],
            b_dense: [[T; $dense_stages]; $stages + $extra_stages],
            // For interpolation caching
            cached_step_num: usize,
            cont: [$crate::ode::SMatrix<T, R, C>; $dense_stages], // Interpolation polynomial coefficients

            // Settings
            pub rtol: T,
            pub atol: T,
            pub h_max: T,
            pub h_min: T,
            pub max_steps: usize,
            pub max_rejects: usize,
            pub safety_factor: T,
            pub min_scale: T,
            pub max_scale: T,
            
            // Iteration tracking
            reject: bool,
            n_stiff: usize,

            // Statistic Tracking
            pub evals: usize,
            pub steps: usize,
            pub rejected_steps: usize,
            pub accepted_steps: usize,

            // Status
            status: $crate::SolverStatus<T, R, C, E>,
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::traits::EventData> Default for $name<T, R, C, E> {
            fn default() -> Self {
                // Initialize k vectors with zeros
                let k: [$crate::ode::SMatrix<T, R, C>; $stages + $extra_stages] = [$crate::ode::SMatrix::<T, R, C>::zeros(); $stages + $extra_stages];

                // Initialize interpolation coefficient storage
                let cont: [$crate::ode::SMatrix<T, R, C>; $dense_stages] = [$crate::ode::SMatrix::<T, R, C>::zeros(); $dense_stages];

                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                
                // Handle the 2D array for b, where first row is higher order and second row is lower order
                let b_higher: [T; $stages] = $b[0].map(|x| T::from_f64(x).unwrap());
                let b_lower: [T; $stages] = $b[1].map(|x| T::from_f64(x).unwrap());
                
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                // Convert interpolation coefficients
                let a_dense_t: [[T; $stages]; $extra_stages] = $a_dense.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let c_dense_t: [T; $extra_stages] = $c_dense.map(|x| T::from_f64(x).unwrap());
                let b_dense_t: [[T; $dense_stages]; $stages + $extra_stages] = 
                    $b_dense.map(|row| row.map(|x| T::from_f64(x).unwrap()));

                $name {
                    h0: T::from_f64(0.1).unwrap(),
                    h: T::from_f64(0.1).unwrap(),
                    t: T::from_f64(0.0).unwrap(),
                    y: $crate::ode::SMatrix::<T, R, C>::zeros(),
                    dydt: $crate::ode::SMatrix::<T, R, C>::zeros(),
                    t_prev: T::from_f64(0.0).unwrap(),
                    y_prev: $crate::ode::SMatrix::<T, R, C>::zeros(),
                    dydt_prev: $crate::ode::SMatrix::<T, R, C>::zeros(),
                    k,
                    a: a_t,
                    b_higher: b_higher,
                    b_lower: b_lower,
                    c: c_t,
                    a_dense: a_dense_t,
                    c_dense: c_dense_t,
                    b_dense: b_dense_t,
                    cached_step_num: 0,
                    cont,
                    rtol: T::from_f64(1.0e-6).unwrap(),
                    atol: T::from_f64(1.0e-6).unwrap(),
                    h_max: T::infinity(),
                    h_min: T::from_f64(0.0).unwrap(),
                    max_steps: 10000,
                    max_rejects: 100,
                    safety_factor: T::from_f64(0.9).unwrap(),
                    min_scale: T::from_f64(0.2).unwrap(),
                    max_scale: T::from_f64(10.0).unwrap(),
                    reject: false,
                    n_stiff: 0,
                    evals: 0,
                    steps: 0,
                    rejected_steps: 0,
                    accepted_steps: 0,
                    status: $crate::SolverStatus::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::traits::EventData> $crate::Solver<T, R, C, E> for $name<T, R, C, E> {
            fn init<F>(&mut self, system: &F, t0: T, tf: T, y: &$crate::ode::SMatrix<T, R, C>) -> Result<(), $crate::SolverStatus<T, R, C, E>>
            where
                F: $crate::System<T, R, C, E>,
            {
                // Check bounds
                match $crate::solvers::utils::validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf) {
                    Ok(h0) => self.h = h0,
                    Err(status) => return Err(status),
                }

                // Initialize Statistics
                self.evals = 0;
                self.steps = 0;
                self.rejected_steps = 0;
                self.accepted_steps = 0;
                self.reject = false;
                self.n_stiff = 0;
                self.cached_step_num = 0;

                // Initialize State
                self.t = t0;
                self.y = y.clone();
                system.diff(t0, y, &mut self.dydt);
                self.evals += 1;

                // Initialize previous state
                self.t_prev = t0;
                self.y_prev = y.clone();
                self.dydt_prev = self.dydt.clone();

                // Initialize Status
                self.status = $crate::SolverStatus::Initialized;

                Ok(())
            }

            fn step<F>(&mut self, system: &F)
            where
                F: $crate::System<T, R, C, E>,
            {
                // Make sure step size isn't too small
                if self.h.abs() < T::default_epsilon() {
                    self.status = $crate::SolverStatus::StepSize(self.t, self.y.clone());
                    return;
                }

                // Check if max steps has been reached
                if self.steps >= self.max_steps {
                    self.status = $crate::SolverStatus::MaxSteps(self.t, self.y.clone());
                    return;
                }

                // Save k[0] as the current derivative
                self.k[0] = self.dydt.clone();
                
                // Compute stages
                for i in 1..$stages {
                    let mut y_stage = self.y.clone();
                    
                    for j in 0..i {
                        y_stage += self.k[j] * (self.a[i][j] * self.h);
                    }
                    
                    system.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
                }
                self.evals += $stages - 1; // We already have k[0]
                
                // Compute higher order solution
                let mut y_high = self.y.clone();
                for i in 0..$stages {
                    y_high += self.k[i] * (self.b_higher[i] * self.h);
                }
                
                // Compute lower order solution for error estimation
                let mut y_low = self.y.clone();
                for i in 0..$stages {
                    y_low += self.k[i] * (self.b_lower[i] * self.h);
                }
                
                // Compute error estimate
                let err = y_high - y_low;
                
                // Calculate error norm using WRMS (weighted root mean square) norm
                let mut err_norm: T = T::zero();
                
                // Iterate through matrix elements
                for r in 0..R {
                    for c in 0..C {
                        let tol = self.atol + self.rtol * self.y[(r, c)].abs().max(y_high[(r, c)].abs());
                        err_norm = err_norm.max((err[(r, c)] / tol).abs());
                    }
                }
                
                // Determine if step is accepted
                if err_norm <= T::one() {
                    // Log previous state
                    self.t_prev = self.t;
                    self.y_prev = self.y.clone();
                    self.dydt_prev = self.dydt.clone();

                    if self.reject {
                        // Not rejected this time
                        self.n_stiff = 0;
                        self.reject = false;
                        self.status = $crate::SolverStatus::Solving;
                    }
                    
                    // Update state with the higher-order solution
                    self.t += self.h;
                    self.y = y_high;
                    system.diff(self.t, &self.y, &mut self.dydt);
                    self.evals += 1;

                    // Update statistics
                    self.steps += 1;
                    self.accepted_steps += 1;
                } else {
                    // Step rejected
                    self.reject = true;
                    self.rejected_steps += 1;
                    self.status = $crate::SolverStatus::RejectedStep;
                    self.n_stiff += 1;
                    
                    // Check for stiffness
                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::SolverStatus::Stiffness(self.t, self.y.clone());
                        return;
                    }
                }
                
                // Calculate new step size
                let order = T::from_usize($order).unwrap();
                let err_order = T::one() / order;
                
                // Standard step size controller formula
                let scale = self.safety_factor * err_norm.powf(-err_order);
                
                // Apply constraints to step size changes
                let scale = scale.max(self.min_scale).min(self.max_scale);
                
                // Update step size
                self.h *= scale;
                
                // Ensure step size is within bounds
                self.h = $crate::solvers::utils::constrain_step_size(self.h, self.h_min, self.h_max);
            }

            fn interpolate<F>(&mut self, system: &F, t_interp: T) -> $crate::ode::SMatrix<T, R, C>
            where
                F: $crate::System<T, R, C, E>
            {
                // Calculate the normalized distance within the step [0, 1]
                let u = (t_interp - self.t_prev) / (self.t - self.t_prev);

                // Compute interpolation coefficients if needed (if we moved to a new step)
                if self.cached_step_num != self.steps {
                    // Compute extra stages for interpolation
                    for i in 0..$extra_stages {
                        let mut y_stage = self.y_prev.clone();
                        // Sum over the main and previous extra stages
                        for j in 0..$stages {
                            y_stage += self.k[j] * (self.a_dense[i][j] * self.h);
                        }
                        
                        // Extra stages might depend on previous extra stages
                        for j in 0..i {
                            y_stage += self.k[$stages + j] * (self.a_dense[i][$stages + j] * self.h);
                        }
                        
                        system.diff(self.t_prev + self.c_dense[i] * self.h, &y_stage, &mut self.k[$stages + i]);
                    }
                    self.evals += $extra_stages;
                    
                    // Clear the interpolation coefficients
                    for i in 0..$dense_stages {
                        self.cont[i] = $crate::ode::SMatrix::<T, R, C>::zeros();
                    }
                    
                    // Compute the coefficients for the interpolation polynomial
                    for i in 0..$stages + $extra_stages {
                        for j in 0..$dense_stages {
                            self.cont[j] += self.k[i] * self.b_dense[i][j];
                        }
                    }
                    
                    // Mark the step as cached
                    self.cached_step_num = self.steps;
                }
                
                // Evaluate the interpolation polynomial at u using Horner's rule
                // P(u) = c0 + u*(c1 + u*(c2 + u*(c3 + ... )))
                let mut result = self.cont[$dense_stages - 1];
                for i in (0..$dense_stages - 1).rev() {
                    result = result * u + self.cont[i];
                }
                
                result
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &$crate::ode::SMatrix<T, R, C> {
                &self.y
            }

            fn dydt(&self) -> &$crate::ode::SMatrix<T, R, C> {
                &self.dydt
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &$crate::ode::SMatrix<T, R, C> {
                &self.y_prev
            }

            fn dydt_prev(&self) -> &$crate::ode::SMatrix<T, R, C> {
                &self.dydt_prev
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
                self.accepted_steps
            }

            fn status(&self) -> &$crate::SolverStatus<T, R, C, E> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::SolverStatus<T, R, C, E>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::traits::EventData> $name<T, R, C, E> {
            /// Create a new solver with the specified initial step size
            pub fn new(h0: T) -> Self {
                Self {
                    h0,
                    h: h0,
                    ..Default::default()
                }
            }
            
            /// Get the number of terms in the dense output interpolation polynomial
            pub fn dense_stages(&self) -> usize {
                $dense_stages
            }
            
            /// Set the relative tolerance for error control
            pub fn rtol(mut self, rtol: T) -> Self {
                self.rtol = rtol;
                self
            }
            
            /// Set the absolute tolerance for error control
            pub fn atol(mut self, atol: T) -> Self {
                self.atol = atol;
                self
            }
            
            /// Set the minimum allowed step size
            pub fn h_min(mut self, h_min: T) -> Self {
                self.h_min = h_min;
                self
            }
            
            /// Set the maximum allowed step size
            pub fn h_max(mut self, h_max: T) -> Self {
                self.h_max = h_max;
                self
            }
            
            /// Set the maximum number of steps allowed
            pub fn max_steps(mut self, max_steps: usize) -> Self {
                self.max_steps = max_steps;
                self
            }
            
            /// Set the maximum number of consecutive rejected steps before declaring stiffness
            pub fn max_rejects(mut self, max_rejects: usize) -> Self {
                self.max_rejects = max_rejects;
                self
            }
            
            /// Set the safety factor for step size control (default: 0.9)
            pub fn safety_factor(mut self, safety_factor: T) -> Self {
                self.safety_factor = safety_factor;
                self
            }
            
            /// Set the minimum scale factor for step size changes (default: 0.2)
            pub fn min_scale(mut self, min_scale: T) -> Self {
                self.min_scale = min_scale;
                self
            }
            
            /// Set the maximum scale factor for step size changes (default: 10.0)
            pub fn max_scale(mut self, max_scale: T) -> Self {
                self.max_scale = max_scale;
                self
            }
            
            /// Get the order of the method
            pub fn order(&self) -> usize {
                $order
            }

            /// Get the number of stages in the method
            pub fn stages(&self) -> usize {
                $stages
            }
        }
    };
}