/// Macro to create a Runge-Kutta solver from a Butcher tableau with fixed-size arrays
///
/// # Arguments
/// 
/// * `name`: Name of the solver struct to create
/// * `doc`: Documentation string for the solver
/// * `a`: Matrix of coefficients for intermediate stages
/// * `b`: Weights for final summation
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the method
/// * `stages`: Number of stages in the method
///
/// # Example
/// 
/// ```
/// use differential_equations::ode::runge_kutta_method;
/// 
/// // Define classical RK4 method
/// runge_kutta_method!(
///     /// Classical 4th Order Runge-Kutta Method
///     name: RK4,
///     a: [[0.0, 0.0, 0.0, 0.0],
///         [0.5, 0.0, 0.0, 0.0],
///         [0.0, 0.5, 0.0, 0.0],
///         [0.0, 0.0, 1.0, 0.0]],
///     b: [1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0],
///     c: [0.0, 0.5, 0.5, 1.0],
///     order: 4,
///     stages: 4
/// );
/// ```
/// 
/// # Note on Butcher Tableaus
/// 
/// The `a` matrix is typically a lower triangular matrix with zeros on the diagonal.
/// when creating the `a` matrix for implementation simplicity it is generated as a
/// 2D array with zeros in the upper triangular portion of the matrix. The array size
/// is known at compile time and it is a O(1) operation to access the desired elements.
/// When computing the Runge-Kutta stages only the elements in the lower triangular portion
/// of the matrix and unnessary multiplication by zero is avoided. The Rust compiler is also
/// likely to optimize the array out instead of memory addresses directly.
/// 
#[macro_export]
macro_rules! runge_kutta_method {
    (
        $(#[$attr:meta])*
        name: $name:ident,
        a: $a:expr,
        b: $b:expr,
        c: $c:expr,
        order: $order:expr,
        stages: $stages:expr
        $(,)? // Optional trailing comma
    ) => {

        
        $(#[$attr])*
        #[doc = "\n\n"]
        #[doc = "This solver was automatically generated using the `runge_kutta_method` macro."]
        pub struct $name<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> {
            // Step Size
            pub h: T,

            // Current State
            t: T,
            y: $crate::SMatrix<T, R, C>,

            // Previous State
            t_prev: T,
            y_prev: $crate::SMatrix<T, R, C>,
            dydt_prev: $crate::SMatrix<T, R, C>,

            // Stage values (fixed size arrays of Vectors)
            k: [$crate::SMatrix<T, R, C>; $stages],

            // Constants from Butcher tableau (fixed size arrays)
            a: [[T; $stages]; $stages],
            b: [T; $stages],
            c: [T; $stages],

            // Statistic Tracking
            pub evals: usize,
            pub steps: usize,

            // Status
            status: $crate::ode::SolverStatus<T, R, C, E>,
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> Default for $name<T, R, C, E> {
            fn default() -> Self {
                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let b_t: [T; $stages] = $b.map(|x| T::from_f64(x).unwrap());
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());
                
                $name {
                    h: T::from_f64(0.1).unwrap(),
                    t: T::from_f64(0.0).unwrap(),
                    y: $crate::SMatrix::<T, R, C>::zeros(),
                    t_prev: T::from_f64(0.0).unwrap(),
                    y_prev: $crate::SMatrix::<T, R, C>::zeros(),
                    dydt_prev: $crate::SMatrix::<T, R, C>::zeros(),
                    // Initialize k vectors with zeros
                    k: [$crate::SMatrix::<T, R, C>::zeros(); $stages],
                    // Use the converted Butcher tableau
                    a: a_t,
                    b: b_t,
                    c: c_t,
                    evals: 0,
                    steps: 0,
                    status: $crate::ode::SolverStatus::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> $crate::ode::Solver<T, R, C, E> for $name<T, R, C, E> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &$crate::SMatrix<T, R, C>) -> Result<(), $crate::ode::SolverStatus<T, R, C, E>>
            where 
                F: $crate::ode::ODE<T, R, C, E>
            {
                // Check Bounds
                match $crate::ode::solvers::utils::validate_step_size_parameters(self.h, T::zero(), T::infinity(), t0, tf) {
                    Ok(h) => self.h = h,
                    Err(e) => return Err(e),
                }

                // Initialize Statistics
                self.evals = 0;
                self.steps = 0;

                // Initialize State
                self.t = t0;
                self.y = y.clone();
                ode.diff(t0, y, &mut self.k[0]);

                // Initialize previous state
                self.t_prev = t0;
                self.y_prev = y.clone();
                self.dydt_prev = self.k[0];

                // Initialize Status
                self.status = $crate::ode::SolverStatus::Initialized;

                Ok(())
            }

            fn step<F>(&mut self, ode: &F) 
            where 
                F: $crate::ode::ODE<T, R, C, E>
            {
                // Log previous state
                self.t_prev = self.t;
                self.y_prev = self.y;
                self.dydt_prev = self.k[0];

                // Compute k_0 = f(t, y)
                ode.diff(self.t, &self.y, &mut self.k[0]);

                // Compute stage values
                for i in 1..$stages {
                    // Start with the original y value
                    let mut stage_y = self.y;
                    
                    // Add contribution from previous stages
                    for j in 0..i {
                        stage_y += self.k[j] * (self.a[i][j] * self.h);
                    }
                    
                    // Compute k_i = f(t + c_i*h, stage_y)
                    ode.diff(self.t + self.c[i] * self.h, &stage_y, &mut self.k[i]);
                }
                
                // Compute the final update
                let mut delta_y = $crate::SMatrix::<T, R, C>::zeros();
                for i in 0..$stages {
                    delta_y += self.k[i] * (self.b[i] * self.h);
                }
                
                // Update state
                self.y += delta_y;
                self.t += self.h;

                // Update Statistics
                self.steps += 1;
                self.evals += $stages;
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y
            }

            fn dydt(&self) -> &$crate::SMatrix<T, R, C> {
                &self.k[0]
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y_prev
            }

            fn dydt_prev(&self) -> &$crate::SMatrix<T, R, C> {
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
                0
            }

            fn accepted_steps(&self) -> usize {
                self.steps
            }

            fn status(&self) -> &$crate::ode::SolverStatus<T, R, C, E> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::ode::SolverStatus<T, R, C, E>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> $name<T, R, C, E> {
            /// Create a new solver with the specified step size
            ///
            /// # Arguments
            /// * `h` - Step size
            ///
            /// # Returns
            /// * A new solver instance
            pub fn new(h: T) -> Self {
                $name {
                    h,
                    ..Default::default()
                }
            }
            
            /// Get the order of accuracy of this method
            pub fn order(&self) -> usize {
                $order
            }
            
            /// Get the number of stages in this method
            pub fn stages(&self) -> usize {
                $stages
            }
        }
    };
}

/// Macro to create an adaptive Runge-Kutta solver with embedded error estimation
///
/// # Arguments
/// 
/// * `name`: Name of the solver struct to create
/// * `a`: Matrix of coefficients for intermediate stages
/// * `b`: 2D array where first row is higher order weights, second row is lower order weights
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the method
/// * `stages`: Number of stages in the method
///
/// # Example
/// 
/// ```
/// use differential_equations::ode::adaptive_runge_kutta_method;
/// 
/// // Define RKF45 method
/// adaptive_runge_kutta_method!(
///     /// Runge-Kutta-Fehlberg 4(5) adaptive step size method
///     name: RKF,
///     a: [
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
///         [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
///         [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0],
///         [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]
///     ],
///     b: [
///         [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], // 5th order
///         [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]           // 4th order
///     ],
///     c: [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0],
///     order: 5,
///     stages: 6
/// );
/// ```
/// 
/// # Note on Butcher Tableaus
/// 
/// The `a` matrix is typically a lower triangular matrix with zeros on the diagonal.
/// when creating the `a` matrix for implementation simplicity it is generated as a
/// 2D array with zeros in the upper triangular portion of the matrix. The array size
/// is known at compile time and it is a O(1) operation to access the desired elements.
/// When computing the Runge-Kutta stages only the elements in the lower triangular portion
/// of the matrix and unnessary multiplication by zero is avoided. The Rust compiler is also
/// likely to optimize the array out instead of memory addresses directly.
/// 
/// The `b` matrix is a 2D array where the first row is the higher order weights and the
/// second row is the lower order weights. This is used for embedded error estimation.
/// 
#[macro_export]
macro_rules! adaptive_runge_kutta_method {
    (
        $(#[$attr:meta])*
        name: $name:ident,
        a: $a:expr,
        b: $b:expr,
        c: $c:expr,
        order: $order:expr,
        stages: $stages:expr
        $(,)? // Optional trailing comma
    ) => {
        $(#[$attr])*
        #[doc = "\n\n"]
        #[doc = "This adaptive solver was automatically generated using the `adaptive_runge_kutta_method` macro."]
        pub struct $name<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> {
            // Initial Step Size
            pub h0: T,

            // Current Step Size
            h: T,

            // Current State
            t: T,
            y: $crate::SMatrix<T, R, C>,
            dydt: $crate::SMatrix<T, R, C>,

            // Previous State
            t_prev: T,
            y_prev: $crate::SMatrix<T, R, C>,
            dydt_prev: $crate::SMatrix<T, R, C>,

            // Stage values (fixed size array of Vs)
            k: [$crate::SMatrix<T, R, C>; $stages],

            // Constants from Butcher tableau (fixed size arrays)
            a: [[T; $stages]; $stages],
            b_higher: [T; $stages],
            b_lower: [T; $stages],
            c: [T; $stages],

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
            status: $crate::ode::SolverStatus<T, R, C, E>,
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> Default for $name<T, R, C, E> {
            fn default() -> Self {
                // Initialize k vectors with zeros
                let k: [$crate::SMatrix<T, R, C>; $stages] = [$crate::SMatrix::<T, R, C>::zeros(); $stages];

                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                
                // Handle the 2D array for b, where first row is higher order and second row is lower order
                let b_higher: [T; $stages] = $b[0].map(|x| T::from_f64(x).unwrap());
                let b_lower: [T; $stages] = $b[1].map(|x| T::from_f64(x).unwrap());
                
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                $name {
                    h0: T::from_f64(0.1).unwrap(),
                    h: T::from_f64(0.1).unwrap(),
                    t: T::from_f64(0.0).unwrap(),
                    y: $crate::SMatrix::<T, R, C>::zeros(),
                    dydt: $crate::SMatrix::<T, R, C>::zeros(),
                    t_prev: T::from_f64(0.0).unwrap(),
                    y_prev: $crate::SMatrix::<T, R, C>::zeros(),
                    dydt_prev: $crate::SMatrix::<T, R, C>::zeros(),
                    k,
                    a: a_t,
                    b_higher,         // Higher order (b)
                    b_lower,           // Lower order (b_hat)
                    c: c_t,
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
                    status: $crate::ode::SolverStatus::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> $crate::ode::Solver<T, R, C, E> for $name<T, R, C, E> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &$crate::SMatrix<T, R, C>) -> Result<(), $crate::ode::SolverStatus<T, R, C, E>>
            where
                F: $crate::ode::ODE<T, R, C, E>,
            {
                // Check bounds
                match $crate::ode::solvers::utils::validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf) {
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

                // Initialize State
                self.t = t0;
                self.y = y.clone();
                ode.diff(t0, y, &mut self.dydt);

                // Initialize previous state
                self.t_prev = t0;
                self.y_prev = y.clone();
                self.dydt_prev = self.dydt;

                // Initialize Status
                self.status = $crate::ode::SolverStatus::Initialized;

                Ok(())
            }

            fn step<F>(&mut self, ode: &F)
            where
                F: $crate::ode::ODE<T, R, C, E>,
            {
                // Make sure step size isn't too small
                if self.h.abs() < T::default_epsilon() {
                    self.status = $crate::ode::SolverStatus::StepSize(self.t, self.y.clone());
                    return;
                }

                // Check if max steps has been reached
                if self.steps >= self.max_steps {
                    self.status = $crate::ode::SolverStatus::MaxSteps(self.t, self.y.clone());
                    return;
                }

                // Compute stages
                ode.diff(self.t, &self.y, &mut self.k[0]);
                
                for i in 1..$stages {
                    let mut y_stage = self.y;
                    
                    for j in 0..i {
                        y_stage += self.k[j] * (self.a[i][j] * self.h);
                    }
                    
                    ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
                }
                
                // Compute higher order solution
                let mut y_high = self.y;
                for i in 0..$stages {
                    y_high += self.k[i] * (self.b_higher[i] * self.h);
                }
                
                // Compute lower order solution for error estimation
                let mut y_low = self.y;
                for i in 0..$stages {
                    y_low += self.k[i] * (self.b_lower[i] * self.h);
                }
                
                // Compute error estimate
                let err = y_high - y_low;
                
                // Calculate error norm
                // Using WRMS (weighted root mean square) norm
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
                    self.y_prev = self.y;
                    self.dydt_prev = self.dydt;

                    if self.reject {
                        // Not rejected this time
                        self.n_stiff = 0;
                        self.reject = false;
                        self.status = $crate::ode::SolverStatus::Solving;
                    }
                    
                    // Update state with the higher-order solution
                    self.t += self.h;
                    self.y = y_high;
                    ode.diff(self.t, &self.y, &mut self.dydt);

                    // Update statistics
                    self.steps += 1;
                    self.accepted_steps += 1;
                    self.evals += $stages + 1;
                } else {
                    // Step rejected
                    self.reject = true;
                    self.rejected_steps += 1;
                    self.evals += $stages;
                    self.status = $crate::ode::SolverStatus::RejectedStep;
                    self.n_stiff += 1;
                    
                    // Check for stiffness
                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::ode::SolverStatus::Stiffness(self.t, self.y.clone());
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
                self.h = $crate::ode::solvers::utils::constrain_step_size(self.h, self.h_min, self.h_max);
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y
            }

            fn dydt(&self) -> &$crate::SMatrix<T, R, C> {
                &self.dydt
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y_prev
            }

            fn dydt_prev(&self) -> &$crate::SMatrix<T, R, C> {
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

            fn status(&self) -> &$crate::ode::SolverStatus<T, R, C, E> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::ode::SolverStatus<T, R, C, E>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, E: $crate::ode::EventData> $name<T, R, C, E> {
            /// Create a new solver with the specified initial step size
            pub fn new(h0: T) -> Self {
                Self {
                    h0,
                    h: h0,
                    ..Default::default()
                }
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