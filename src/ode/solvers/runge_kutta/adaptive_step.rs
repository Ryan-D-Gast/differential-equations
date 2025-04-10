//! Adaptive step size Runge-Kutta methods without integrated dense output via cubic Hermite interpolation.

use crate::adaptive_runge_kutta_method;

adaptive_runge_kutta_method!(
    /// Runge-Kutta-Fehlberg 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The RKF45 method is one of the most widely used adaptive step size methods due to
    /// its excellent balance of efficiency and accuracy.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/4    | 1/4
    /// 3/8    | 3/32         9/32
    /// 12/13  | 1932/2197    -7200/2197  7296/2197
    /// 1      | 439/216      -8          3680/513    -845/4104
    /// 1/2    | -8/27        2           -3544/2565  1859/4104   -11/40
    /// -----------------------------------------------------------------------
    ///        | 16/135       0           6656/12825  28561/56430 -9/50       2/55    (5th order)
    ///        | 25/216       0           1408/2565   2197/4104   -1/5        0       (4th order)
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method#CITEREFFehlberg1969)
    name: RKF,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]
    ],
    b: [
        [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], // 5th order
        [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]           // 4th order
    ],
    c: [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0],
    order: 5,
    stages: 6
);

adaptive_runge_kutta_method!(
    /// Cash-Karp 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The Cash-Karp method is a variant of the Runge-Kutta-Fehlberg method that uses
    /// different coefficients to achieve a more efficient and accurate solution.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 3/5    | 3/10         -9/10       6/5
    /// 1      | -11/54       5/2         -70/27      35/27
    /// 7/8    | 1631/55296   175/512     575/13824   44275/110592 253/4096
    /// ------------------------------------------------------------------------------------
    ///        | 37/378       0           250/621     125/594     0           512/1771  (5th order)
    ///        | 2825/27648   0           18575/48384 13525/55296 277/14336   1/4       (4th order)
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method)
    name: CashKarp,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/10.0, -9.0/10.0, 6.0/5.0, 0.0, 0.0, 0.0],
        [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0, 0.0, 0.0],
        [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, 0.0]
    ],
    b: [
        [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0], // 5th order
        [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0] // 4th order
    ],
    c: [0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0],
    order: 5,
    stages: 6
);

/// Macro to create an adaptive Runge-Kutta solver with embedded error estimation
/// and interpolation vs cubic Hermite interpolation.
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
/// use differential_equations::adaptive_runge_kutta_method;
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
        pub struct $name<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::ode::CallBackData> {
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
            steps: usize, // Number of steps taken

            // Status
            status: $crate::ode::SolverStatus<T, R, C, D>,
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::ode::CallBackData> Default for $name<T, R, C, D> {
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
                    b_higher, // Higher order (b)
                    b_lower,  // Lower order (b_hat)
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
                    steps: 0,
                    status: $crate::ode::SolverStatus::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::ode::CallBackData> $crate::ode::Solver<T, R, C, D> for $name<T, R, C, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &$crate::SMatrix<T, R, C>) -> Result<usize, $crate::ode::SolverError<T, R, C>>
            where
                F: $crate::ode::ODE<T, R, C, D>,
            {
                // Check bounds
                match $crate::ode::solvers::utils::validate_step_size_parameters::<T, R, C, D>(self.h0, self.h_min, self.h_max, t0, tf) {
                    Ok(h0) => self.h = h0,
                    Err(status) => return Err(status),
                }

                // Initialize Statistics
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

                Ok(1)
            }

            fn step<F>(&mut self, ode: &F) -> Result<usize, $crate::ode::SolverError<T, R, C>>
            where
                F: $crate::ode::ODE<T, R, C, D>,
            {
                // Make sure step size isn't too small
                if self.h.abs() < T::default_epsilon() {
                    self.status = $crate::ode::SolverStatus::Error($crate::ode::SolverError::StepSize {
                        t: self.t, 
                        y: self.y
                    });
                    return Err($crate::ode::SolverError::StepSize {
                        t: self.t,
                        y: self.y
                    });
                }

                // Check if max steps has been reached
                if self.steps >= self.max_steps {
                    self.status = $crate::ode::SolverStatus::Error($crate::ode::SolverError::MaxSteps {
                        t: self.t, 
                        y: self.y
                    });
                    return Err($crate::ode::SolverError::MaxSteps {
                        t: self.t,
                        y: self.y
                    });
                }
                self.steps += 1;

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

                let mut evals = 0;

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
                    evals += $stages + 1;
                } else {
                    // Step rejected
                    self.reject = true;

                    evals += $stages;
                    self.status = $crate::ode::SolverStatus::RejectedStep;
                    self.n_stiff += 1;

                    // Check for stiffness
                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::ode::SolverStatus::Error($crate::ode::SolverError::Stiffness {
                            t: self.t, y: self.y
                        });
                        return Err($crate::ode::SolverError::Stiffness {
                            t: self.t, y: self.y
                        });
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
                Ok(evals)
            }

            fn interpolate(&mut self, t_interp: T) -> Result<$crate::SMatrix<T, R, C>, $crate::interpolate::InterpolationError<T, R, C>> {
                // Check if t is within bounds
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::interpolate::InterpolationError::OutOfBounds {
                        t_interp, 
                        t_prev: self.t_prev, 
                        t_curr: self.t
                    });
                }

                // Compute the interpolated value using cubic Hermite interpolation
                let y_interp = $crate::interpolate::cubic_hermite_interpolate(self.t_prev, self.t, &self.y_prev, &self.y, &self.dydt_prev, &self.dydt, t_interp);

                Ok(y_interp)
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &$crate::SMatrix<T, R, C> {
                &self.y_prev
            }

            fn h(&self) -> T {
                self.h
            }

            fn set_h(&mut self, h: T) {
                self.h = h;
            }

            fn status(&self) -> &$crate::ode::SolverStatus<T, R, C, D> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::ode::SolverStatus<T, R, C, D>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::ode::CallBackData> $name<T, R, C, D> {
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
