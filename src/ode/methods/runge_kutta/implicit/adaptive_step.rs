//! Adaptive step size implicit Runge-Kutta methods for solving ordinary differential equations.

/// Macro to create an adaptive implicit Runge-Kutta solver from a Butcher tableau.
///
/// This macro generates the necessary struct and trait implementations for an adaptive-step
/// implicit Runge-Kutta method. It uses a simple fixed-point iteration to solve the
/// implicit stage equations and estimates the error by comparing the result from the
/// primary `b` weights with a secondary set of weights `b_hat`.
///
/// # Arguments
///
/// * `name`: Name of the solver struct to create
/// * `a`: Matrix of coefficients for intermediate stages (can be non-zero on diagonal/upper triangle)
/// * `b`: 2D array where the first row is the primary weights (`b`) and the second row is the secondary weights (`b_hat`) for error estimation.
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the primary method (used for step size control)
/// * `stages`: Number of stages in the method
///
/// # Note on Solver and Error Estimation
/// - The implicit stage equations `k_i = f(t_n + c_i*h, y_n + h * sum(a_{ij}*k_j))` are solved
///   using fixed-point iteration. This may fail to converge for stiff problems unless `h`
///   is sufficiently small (`h * L < 1`).
/// - Error estimation uses the difference between solutions computed with `b` and `b_hat`.
///   The validity of `b_hat` as an error estimator depends on the specific method's tableau.
///   For methods like Gauss-Legendre, this might not be the standard approach.
///
/// # Example (Illustrative - Requires a valid tableau with error estimator)
/// ```rust
/// // Assuming a hypothetical 2-stage, 2nd order implicit method with error estimator
/// /*
/// use differential_equations::adaptive_implicit_runge_kutta_method;
/// adaptive_implicit_runge_kutta_method!(
///     name: AdaptiveImplicitExample,
///     a: [[0.5, 0.0], [0.5, 0.5]], // Example 'a' matrix
///     b: [
///         [0.5, 0.5], // Primary weights (e.g., order 2)
///         [1.0, 0.0]  // Secondary weights (e.g., order 1)
///     ],
///     c: [0.5, 1.0],
///     order: 2,
///     stages: 2
/// );
/// */
/// ```
#[macro_export]
macro_rules! adaptive_implicit_runge_kutta_method {
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
        #[doc = "This adaptive implicit solver was automatically generated using the `adaptive_implicit_runge_kutta_method` macro."]
        #[doc = " It uses fixed-point iteration and embedded error estimation (via b/b_hat vectors)."]
        pub struct $name<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> {
            // Initial Step Size
            pub h0: T,
            // Current Step Size
            h: T,

            // Current State
            t: T,
            y: V,
            dydt: V, // Derivative at t

            // Previous State
            t_prev: T,
            y_prev: V,
            dydt_prev: V, // Derivative at t_prev

            // Stage derivatives (k_i)
            k: [V; $stages],
            // Temporary storage for stage values during iteration
            y_stage: [V; $stages],
            k_new: [V; $stages],

            // Constants from Butcher tableau (fixed size arrays)
            a: [[T; $stages]; $stages],
            b_higher: [T; $stages], // Primary weights (b)
            b_lower: [T; $stages],  // Secondary weights (b_hat) for error estimation
            c: [T; $stages],

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
            pub max_iter: usize, // Max iterations for fixed-point solver
            pub tol: T,          // Tolerance for fixed-point solver convergence

            // Iteration tracking & Status
            reject: bool,
            n_stiff: usize,
            steps: usize,
            nfcn: usize, // Total function evaluations
            naccept: usize,
            nreject: usize,
            status: $crate::Status<T, V, D>,
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> Default for $name<T, V, D> {
            fn default() -> Self {
                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let b_higher_t: [T; $stages] = $b[0].map(|x| T::from_f64(x).unwrap());
                let b_lower_t: [T; $stages] = $b[1].map(|x| T::from_f64(x).unwrap());
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                $name {
                    h0: T::zero(), // Indicate auto-calculation
                    h: T::zero(),
                    t: T::zero(),
                    y: V::zeros(),
                    dydt: V::zeros(),
                    t_prev: T::zero(),
                    y_prev: V::zeros(),
                    dydt_prev: V::zeros(),
                    k: [V::zeros(); $stages],
                    y_stage: [V::zeros(); $stages],
                    k_new: [V::zeros(); $stages],
                    a: a_t,
                    b_higher: b_higher_t,
                    b_lower: b_lower_t,
                    c: c_t,
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
                    // Status
                    reject: false,
                    n_stiff: 0,
                    steps: 0,
                    nfcn: 0,
                    naccept: 0,
                    nreject: 0,
                    status: $crate::Status::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::ode::NumericalMethod<T, V, D> for $name<T, V, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<usize, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>,
            {
                // Calculate initial derivative f(t0, y0)
                let mut initial_dydt = V::zeros();
                ode.diff(t0, y0, &mut initial_dydt);
                self.nfcn = 1;

                // If h0 is zero calculate h0 using initial derivative
                if self.h0 == T::zero() {
                    // Use a simpler h_init for implicit, maybe based on explicit Euler step error
                    // Or adapt the existing h_init if possible. Using explicit version for now.
                     self.h0 = $crate::ode::methods::h_init(ode, t0, tf, y0, $order, self.rtol, self.atol, self.h_min, self.h_max);
                }

                // Check bounds
                self.h = $crate::utils::validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf)?;

                // Initialize Statistics
                self.reject = false;
                self.n_stiff = 0;
                self.steps = 0;
                self.naccept = 0;
                self.nreject = 0;

                // Initialize State
                self.t = t0;
                self.y = *y0;
                self.dydt = initial_dydt; // Store f(t0, y0)

                // Initialize previous state (same as current initially)
                self.t_prev = t0;
                self.y_prev = *y0;
                self.dydt_prev = initial_dydt;

                // Initialize Status
                self.status = $crate::Status::Initialized;

                Ok(self.nfcn)
            }

            fn step<F>(&mut self, ode: &F) -> Result<usize, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>,
            {
                let mut evals_step = 0;

                // Check step size validity
                if self.h.abs() < self.h_min || self.h.abs() < T::default_epsilon() {
                    self.status = $crate::Status::Error($crate::Error::StepSize { t: self.t, y: self.y });
                    return Err($crate::Error::StepSize { t: self.t, y: self.y });
                }

                // Check max steps
                if self.steps >= self.max_steps {
                    self.status = $crate::Status::Error($crate::Error::MaxSteps { t: self.t, y: self.y });
                    return Err($crate::Error::MaxSteps { t: self.t, y: self.y });
                }
                self.steps += 1;

                // --- Fixed-Point Iteration for stage derivatives k_i ---
                // Initial guess: k_i^{(0)} = f(t_n, y_n) (stored in self.dydt)
                for i in 0..$stages {
                    self.k[i] = self.dydt;
                }

                let mut converged = false;
                for _iter in 0..self.max_iter {
                    let mut max_diff_sq = T::zero();

                    // Calculate next iteration k_i^{(m+1)} based on k_j^{(m)}
                    for i in 0..$stages {
                        // Calculate stage value y_stage = y_n + h * sum(a_ij * k_j^{(m)})
                        self.y_stage[i] = self.y;
                        for j in 0..$stages {
                            self.y_stage[i] += self.k[j] * (self.a[i][j] * self.h);
                        }

                        // Evaluate f at stage time and value: f(t_n + c_i*h, y_stage)
                        ode.diff(self.t + self.c[i] * self.h, &self.y_stage[i], &mut self.k_new[i]);
                        evals_step += 1;
                    }

                    // Check convergence: max ||k_new_i - k_i|| < tol
                    for i in 0..$stages {
                        let diff = self.k_new[i] - self.k[i];
                        let mut error_norm_sq = T::zero();
                        for idx in 0..diff.len() {
                            error_norm_sq += diff.get(idx) * diff.get(idx);
                        }
                        max_diff_sq = max_diff_sq.max(error_norm_sq);

                        // Update k for next iteration
                        self.k[i] = self.k_new[i];
                    }

                    if max_diff_sq.sqrt() < self.tol {
                        converged = true;
                        break;
                    }
                } // End fixed-point iteration loop

                if !converged {
                    // Iteration failed to converge, likely need smaller step
                    // Reduce step size significantly and reject
                    self.h *= T::from_f64(0.25).unwrap(); // Or use min_scale?
                    self.h = $crate::utils::constrain_step_size(self.h, self.h_min, self.h_max);
                    self.reject = true;
                    self.n_stiff += 1; // Count as a potential stiffness issue
                    self.nreject += 1;
                    self.nfcn += evals_step;

                    // Check for excessive rejects/stiffness
                    if self.n_stiff >= self.max_rejects {
                         self.status = $crate::Status::Error($crate::Error::Stiffness { t: self.t, y: self.y });
                         return Err($crate::Error::Stiffness { t: self.t, y: self.y });
                    }
                    // Return 0 evaluations for the step itself, but update total nfcn
                    return Ok(0); // Indicate step was rejected, no state change
                }

                // --- Iteration converged, compute solutions and error ---
                // Compute higher order solution (y_high)
                let mut delta_y_high = V::zeros();
                for i in 0..$stages {
                    delta_y_high += self.k[i] * (self.b_higher[i] * self.h);
                }
                let y_high = self.y + delta_y_high;

                // Compute lower order solution (y_low) for error estimation
                let mut delta_y_low = V::zeros();
                for i in 0..$stages {
                    delta_y_low += self.k[i] * (self.b_lower[i] * self.h);
                }
                let y_low = self.y + delta_y_low;

                // Compute error estimate
                let err = y_high - y_low;

                // Calculate error norm (WRMS norm)
                let mut err_norm = T::zero();
                for n in 0..self.y.len() {
                    let scale = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
                    if scale > T::zero() { // Avoid division by zero
                       err_norm = err_norm.max((err.get(n) / scale).abs());
                    }
                }
                // Handle case where norm might be exactly zero
                err_norm = err_norm.max(T::default_epsilon() * T::from_f64(100.0).unwrap());


                // --- Step Control ---
                let order_inv = T::one() / T::from_usize($order).unwrap();
                let mut scale = self.safety_factor * err_norm.powf(-order_inv);
                scale = scale.max(self.min_scale).min(self.max_scale);
                let h_new = self.h * scale;

                // Determine if step is accepted
                if err_norm <= T::one() { // Accept step
                    self.naccept += 1;
                    self.status = $crate::Status::Solving;

                    // Store previous state
                    self.t_prev = self.t;
                    self.y_prev = self.y;
                    self.dydt_prev = self.dydt; // Store f(t_n, y_n)

                    // Update state
                    self.t += self.h;
                    self.y = y_high;

                    // Calculate derivative at the new point for the *next* step's prediction
                    // and for interpolation.
                    ode.diff(self.t, &self.y, &mut self.dydt);
                    evals_step += 1; // Count this evaluation

                    // Reset rejection flags if previously rejected
                    if self.reject {
                        self.n_stiff = 0;
                        self.reject = false;
                    }

                    // Update step size for the *next* step
                    self.h = $crate::utils::constrain_step_size(h_new, self.h_min, self.h_max);

                } else { // Reject step
                    self.nreject += 1;
                    self.status = $crate::Status::RejectedStep;
                    self.reject = true;
                    self.n_stiff += 1;

                    // Check for stiffness
                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::Status::Error($crate::Error::Stiffness { t: self.t, y: self.y });
                        return Err($crate::Error::Stiffness { t: self.t, y: self.y });
                    }

                    // Update step size for the *retry* of the current step
                    // Use the more conservative h_new calculated from the failed step
                    self.h = $crate::utils::constrain_step_size(h_new, self.h_min, self.h_max);

                    // Do not update t, y, t_prev, y_prev, dydt_prev, dydt
                    // Return 0 evaluations for the step itself, but update total nfcn
                    self.nfcn += evals_step;
                    return Ok(0); // Indicate step rejected, state not advanced
                }

                self.nfcn += evals_step;
                Ok(evals_step) // Return evals for this accepted step
            }

            // --- Standard trait methods ---
            fn t(&self) -> T { self.t }
            fn y(&self) -> &V { &self.y }
            fn t_prev(&self) -> T { self.t_prev }
            fn y_prev(&self) -> &V { &self.y_prev }
            fn h(&self) -> T { self.h }
            fn set_h(&mut self, h: T) { self.h = h; } // Allow external setting, but init recalculates h0 if zero
            fn status(&self) -> &$crate::Status<T, V, D> { &self.status }
            fn set_status(&mut self, status: $crate::Status<T, V, D>) { self.status = status; }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::interpolate::Interpolation<T, V> for $name<T, V, D> {
            fn interpolate(&mut self, t_interp: T) -> Result<V, $crate::interpolate::InterpolationError<T>> {
                 if self.t == self.t_prev { // Handle case before first accepted step
                     if t_interp == self.t_prev {
                         return Ok(self.y_prev);
                     } else {
                         return Err($crate::interpolate::InterpolationError::OutOfBounds { t_interp, t_prev: self.t_prev, t_curr: self.t });
                     }
                 }
                // Check if t is within the bounds of the last accepted step
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::interpolate::InterpolationError::OutOfBounds {
                        t_interp,
                        t_prev: self.t_prev,
                        t_curr: self.t });
                }

                // Use cubic Hermite interpolation using derivatives at the start and end of the interval
                // dydt_prev = f(t_prev, y_prev)
                // dydt = f(t, y)
                let y_interp = $crate::interpolate::cubic_hermite_interpolate(
                    self.t_prev, self.t,
                    &self.y_prev, &self.y,
                    &self.dydt_prev, &self.dydt, // Pass derivatives directly
                    t_interp
                );

                Ok(y_interp)
            }
        }

        // --- Builder Pattern Methods ---
        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $name<T, V, D> {
             /// Creates a new solver instance with default settings.
            pub fn new() -> Self {
                Self::default()
            }

            /// Set initial step size. If not set or set to zero, it will be estimated during init.
            pub fn h0(mut self, h0: T) -> Self { self.h0 = h0; self }
            /// Set the relative tolerance for error control.
            pub fn rtol(mut self, rtol: T) -> Self { self.rtol = rtol; self }
            /// Set the absolute tolerance for error control.
            pub fn atol(mut self, atol: T) -> Self { self.atol = atol; self }
            /// Set the minimum allowed step size.
            pub fn h_min(mut self, h_min: T) -> Self { self.h_min = h_min; self }
            /// Set the maximum allowed step size.
            pub fn h_max(mut self, h_max: T) -> Self { self.h_max = h_max; self }
            /// Set the maximum number of steps allowed.
            pub fn max_steps(mut self, max_steps: usize) -> Self { self.max_steps = max_steps; self }
            /// Set the maximum number of consecutive rejected steps or solver failures before declaring stiffness.
            pub fn max_rejects(mut self, max_rejects: usize) -> Self { self.max_rejects = max_rejects; self }
            /// Set the safety factor for step size control (default: 0.9).
            pub fn safety_factor(mut self, safety_factor: T) -> Self { self.safety_factor = safety_factor; self }
            /// Set the minimum scale factor for step size changes (default: 0.2).
            pub fn min_scale(mut self, min_scale: T) -> Self { self.min_scale = min_scale; self }
            /// Set the maximum scale factor for step size changes (default: 10.0).
            pub fn max_scale(mut self, max_scale: T) -> Self { self.max_scale = max_scale; self }
            /// Set the maximum number of fixed-point iterations per step.
            pub fn max_iter(mut self, iter: usize) -> Self { self.max_iter = iter; self }
            /// Set the tolerance for fixed-point iteration convergence.
            pub fn tol(mut self, tol: T) -> Self { self.tol = tol; self }
        }
    };
}

// --- Define Gauss-Legendre Methods ---

// Constants needed
const SQRT3: f64 = 1.732050808;
const SQRT15: f64 = 3.872983346;

adaptive_implicit_runge_kutta_method!(
    /// Gauss-Legendre method of order 4.
    ///
    /// This is a 2-stage implicit Runge-Kutta method.
    /// It is A-stable and self-adjoint.
    /// The error estimation is based on the second 'b' row provided in the tableau,
    /// which corresponds to simplifying order conditions rather than a standard
    /// embedded lower-order method. Use with caution for adaptive stepping.
    ///
    /// Butcher Tableau:
    /// ```text
    /// c1 | a11 a12
    /// c2 | a21 a22
    /// -------------
    ///    | b1  b2    (Order 4)
    ///    | bh1 bh2   (Simplifying conditions)
    ///
    /// c1 = 1/2 - sqrt(3)/6, c2 = 1/2 + sqrt(3)/6
    /// a11 = 1/4, a12 = 1/4 - sqrt(3)/6
    /// a21 = 1/4 + sqrt(3)/6, a22 = 1/4
    /// b1 = 1/2, b2 = 1/2
    /// bh1 = 1/2 + sqrt(3)/2, bh2 = 1/2 - sqrt(3)/2
    /// ```
    name: GaussLegendre4,
    a: [
        [0.25, 0.25 - SQRT3 / 6.0],
        [0.25 + SQRT3 / 6.0, 0.25]
    ],
    b: [
        [0.5, 0.5], // Order 4 weights
        [0.5 + SQRT3 / 2.0, 0.5 - SQRT3 / 2.0] // Simplifying condition weights (used for error estimate)
    ],
    c: [0.5 - SQRT3 / 6.0, 0.5 + SQRT3 / 6.0],
    order: 4,
    stages: 2
);

adaptive_implicit_runge_kutta_method!(
    /// Gauss-Legendre method of order 6.
    ///
    /// This is a 3-stage implicit Runge-Kutta method.
    /// It is A-stable and self-adjoint.
    /// The error estimation is based on the second 'b' row provided in the tableau,
    /// which corresponds to simplifying order conditions rather than a standard
    /// embedded lower-order method. Use with caution for adaptive stepping.
    ///
    /// Butcher Tableau:
    /// ```text
    /// c1 | a11 a12 a13
    /// c2 | a21 a22 a23
    /// c3 | a31 a32 a33
    /// -----------------
    ///    | b1  b2  b3   (Order 6)
    ///    | bh1 bh2 bh3  (Simplifying conditions)
    ///
    /// c1 = 1/2 - sqrt(15)/10, c2 = 1/2, c3 = 1/2 + sqrt(15)/10
    /// a11 = 5/36, a12 = 2/9 - sqrt(15)/15, a13 = 5/36 - sqrt(15)/30
    /// a21 = 5/36 + sqrt(15)/24, a22 = 2/9, a23 = 5/36 - sqrt(15)/24
    /// a31 = 5/36 + sqrt(15)/30, a32 = 2/9 + sqrt(15)/15, a33 = 5/36
    /// b1 = 5/18, b2 = 4/9, b3 = 5/18
    /// bh1 = -5/6, bh2 = 8/3, bh3 = -5/6
    /// ```
    name: GaussLegendre6,
    a: [
        [5.0/36.0, 2.0/9.0 - SQRT15/15.0, 5.0/36.0 - SQRT15/30.0],
        [5.0/36.0 + SQRT15/24.0, 2.0/9.0, 5.0/36.0 - SQRT15/24.0],
        [5.0/36.0 + SQRT15/30.0, 2.0/9.0 + SQRT15/15.0, 5.0/36.0]
    ],
    b: [
        [5.0/18.0, 4.0/9.0, 5.0/18.0], // Order 6 weights
        [-5.0/6.0, 8.0/3.0, -5.0/6.0]  // Simplifying condition weights (used for error estimate)
    ],
    c: [0.5 - SQRT15/10.0, 0.5, 0.5 + SQRT15/10.0],
    order: 6,
    stages: 3
);
