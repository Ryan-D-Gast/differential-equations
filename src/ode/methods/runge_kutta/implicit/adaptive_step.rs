//! Adaptive step size implicit Runge-Kutta methods for solving ordinary differential equations.

/// Macro to create an adaptive implicit Runge-Kutta solver from a Butcher tableau.
///
/// This macro generates the necessary struct and trait implementations for an adaptive-step
/// implicit Runge-Kutta method. It uses Newton's iteration to solve the
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
///   using Newton's iteration. This requires the ODE system to provide its Jacobian.
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
        #[doc = " It uses Newton iteration and embedded error estimation (via b/b_hat vectors)."]
        #[doc = " The ODE system itself must provide the Jacobian via the `ODE` trait if `use_analytical_jacobian` is true (default)."]
        #[doc = " Otherwise, finite differences are used to approximate the Jacobian."]
        pub struct $name<
            T: $crate::traits::Real,
            V: $crate::traits::State<T>,
            D: $crate::traits::CallBackData,
        > {
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
            f_at_stages: [V; $stages], // Stores f(t_stage, y_stage) during Newton iteration

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
            pub max_iter: usize, // Max iterations for Newton solver
            pub tol: T,          // Tolerance for Newton solver convergence
            fd_epsilon_sqrt: T, // Stores sqrt(machine_epsilon) for FD

            // Iteration tracking & Status
            reject: bool,
            n_stiff: usize,
            steps: usize,
            status: $crate::Status<T, V, D>,

            // --- Jacobian and Newton Solver Data ---
            jacobian_matrix: nalgebra::DMatrix<T>, // Jacobian of f: J(t,y)
            newton_matrix: nalgebra::DMatrix<T>,   // Matrix for Newton system (M)
            rhs_newton: nalgebra::DVector<T>,      // RHS vector for Newton system (-phi)
            delta_k_vec: nalgebra::DVector<T>,     // Solution of Newton system (delta_k)
        }

        impl<
            T: $crate::traits::Real,
            V: $crate::traits::State<T>,
            D: $crate::traits::CallBackData,
        > Default for $name<T, V, D> {
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
                    f_at_stages: [V::zeros(); $stages],
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
                    fd_epsilon_sqrt: T::zero(),
                    // Status
                    reject: false,
                    n_stiff: 0,
                    steps: 0,
                    status: $crate::Status::Uninitialized,
                    // Initialize nalgebra structures (empty, to be sized in init)
                    jacobian_matrix: nalgebra::DMatrix::zeros(0, 0),
                    newton_matrix: nalgebra::DMatrix::zeros(0, 0),
                    rhs_newton: nalgebra::DVector::zeros(0),
                    delta_k_vec: nalgebra::DVector::zeros(0),
                }
            }
        }

        impl<
            T: $crate::traits::Real,
            V: $crate::traits::State<T>,
            D: $crate::traits::CallBackData,
        > $crate::ode::NumericalMethod<T, V, D> for $name<T, V, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>, // ODE trait now includes Jacobian
            {
                let mut evals = $crate::alias::Evals::new();

                // Calculate initial derivative f(t0, y0)
                let mut initial_dydt = V::zeros();
                ode.diff(t0, y0, &mut initial_dydt);
                evals.fcn += 1;

                // If h0 is zero calculate h0 using initial derivative
                if self.h0 == T::zero() {
                    self.h0 = $crate::ode::methods::h_init(ode, t0, tf, y0, $order, self.rtol, self.atol, self.h_min, self.h_max);
                }

                // Check bounds
                self.h = $crate::utils::validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf)?;

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
                self.status = $crate::Status::Initialized;

                // Initialize Jacobian and Newton-related matrices/vectors with correct dimensions
                let dim = y0.len();
                self.jacobian_matrix = nalgebra::DMatrix::zeros(dim, dim);
                let newton_system_size = $stages * dim;
                self.newton_matrix = nalgebra::DMatrix::zeros(newton_system_size, newton_system_size);
                self.rhs_newton = nalgebra::DVector::zeros(newton_system_size);
                self.delta_k_vec = nalgebra::DVector::zeros(newton_system_size);
                self.f_at_stages = [V::zeros(); $stages];

                Ok(evals)
            }

            fn step<F>(&mut self, ode: &F) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>, // ODE trait now includes Jacobian
            {
                let mut evals = $crate::alias::Evals::new();
                let dim = self.y.len();

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

                // --- Newton Iteration for stage derivatives k_i ---
                // Initial guess for k_i: k_i^(0) = f(t_n, y_n) (stored in self.dydt)
                for i in 0..$stages {
                    self.k[i] = self.dydt;
                }

                // Calculate Jacobian J_n = df/dy(t_n, y_n) once per step attempt
                ode.jacobian(self.t, &self.y, &mut self.jacobian_matrix);
                evals.jac += 1;

                let mut converged = false;
                for _iter in 0..self.max_iter {
                    // 1. Compute residual phi(K_current) and store -phi in rhs_newton
                    for i in 0..$stages {
                        self.y_stage[i] = self.y; // y_n
                        for j in 0..$stages {
                            self.y_stage[i] += self.k[j] * (self.a[i][j] * self.h);
                        }

                        ode.diff(self.t + self.c[i] * self.h, &self.y_stage[i], &mut self.f_at_stages[i]);
                        evals.fcn += 1;

                        for row_idx in 0..dim {
                            self.rhs_newton[i * dim + row_idx] = self.f_at_stages[i].get(row_idx) - self.k[i].get(row_idx);
                        }
                    }

                    // 2. Form Newton matrix M
                    for i in 0..$stages { // block row index
                        for l in 0..$stages { // block column index
                            let scale_factor = -self.h * self.a[i][l];
                            for r in 0..dim { // row index within the block
                                for c_col in 0..dim { // column index within the block (renamed from c to avoid conflict)
                                    // Direct assignment to the element in newton_matrix
                                    self.newton_matrix[(i * dim + r, l * dim + c_col)] = 
                                        self.jacobian_matrix[(r, c_col)] * scale_factor;
                                }
                            }

                            if i == l { // If it's a diagonal block, add Identity
                                for d_idx in 0..dim { // index for the diagonal of the block
                                    self.newton_matrix[(i * dim + d_idx, l * dim + d_idx)] += T::one();
                                }
                            }
                        }
                    }

                    // 3. Solve M * delta_k_vec = rhs_newton
                    let lu_decomp = nalgebra::LU::new(self.newton_matrix.clone());
                    if let Some(solution) = lu_decomp.solve(&self.rhs_newton) {
                        self.delta_k_vec.copy_from(&solution);
                    } else {
                        converged = false;
                        break;
                    }

                    // 4. Update K: self.k[i] += delta_k_vec_i
                    let mut norm_delta_k_sq = T::zero();
                    for i in 0..$stages {
                        for row_idx in 0..dim {
                            let delta_val = self.delta_k_vec[i * dim + row_idx];
                            let current_val = self.k[i].get(row_idx);
                            self.k[i].set(row_idx, current_val + delta_val);
                            norm_delta_k_sq += delta_val * delta_val;
                        }
                    }

                    // 5. Check convergence: ||delta_k_vec|| < self.tol
                    if norm_delta_k_sq < self.tol * self.tol {
                        converged = true;
                        break;
                    }
                }

                if !converged {
                    self.h *= T::from_f64(0.25).unwrap();
                    self.h = $crate::utils::constrain_step_size(self.h, self.h_min, self.h_max);
                    self.reject = true;
                    self.n_stiff += 1;

                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::Status::Error($crate::Error::Stiffness { t: self.t, y: self.y });
                        return Err($crate::Error::Stiffness { t: self.t, y: self.y });
                    }
                    return Ok(evals);
                }

                // --- Iteration converged, compute solutions and error ---
                let mut delta_y_high = V::zeros();
                for i in 0..$stages {
                    delta_y_high += self.k[i] * (self.b_higher[i] * self.h);
                }
                let y_high = self.y + delta_y_high;

                let mut delta_y_low = V::zeros();
                for i in 0..$stages {
                    delta_y_low += self.k[i] * (self.b_lower[i] * self.h);
                }
                let y_low = self.y + delta_y_low;

                let err = y_high - y_low;

                let mut err_norm = T::zero();
                for n in 0..self.y.len() {
                    let scale = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
                    if scale > T::zero() {
                        err_norm = err_norm.max((err.get(n) / scale).abs());
                    }
                }
                err_norm = err_norm.max(T::default_epsilon() * T::from_f64(100.0).unwrap());

                let order_inv = T::one() / T::from_usize($order).unwrap();
                let mut scale = self.safety_factor * err_norm.powf(-order_inv);
                scale = scale.max(self.min_scale).min(self.max_scale);
                let h_new = self.h * scale;

                if err_norm <= T::one() {
                    self.status = $crate::Status::Solving;

                    self.t_prev = self.t;
                    self.y_prev = self.y;
                    self.dydt_prev = self.dydt;

                    self.t += self.h;
                    self.y = y_high;

                    ode.diff(self.t, &self.y, &mut self.dydt);
                    evals.fcn += 1;

                    if self.reject {
                        self.n_stiff = 0;
                        self.reject = false;
                    }

                    self.h = $crate::utils::constrain_step_size(h_new, self.h_min, self.h_max);
                } else {
                    self.status = $crate::Status::RejectedStep;
                    self.reject = true;
                    self.n_stiff += 1;

                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::Status::Error($crate::Error::Stiffness { t: self.t, y: self.y });
                        return Err($crate::Error::Stiffness { t: self.t, y: self.y });
                    }

                    self.h = $crate::utils::constrain_step_size(h_new, self.h_min, self.h_max);
                    return Ok(evals);
                }

                Ok(evals)
            }

            fn t(&self) -> T { self.t }
            fn y(&self) -> &V { &self.y }
            fn t_prev(&self) -> T { self.t_prev }
            fn y_prev(&self) -> &V { &self.y_prev }
            fn h(&self) -> T { self.h }
            fn set_h(&mut self, h: T) { self.h = h; }
            fn status(&self) -> &$crate::Status<T, V, D> { &self.status }
            fn set_status(&mut self, status: $crate::Status<T, V, D>) { self.status = status; }
        }

        impl<
            T: $crate::traits::Real,
            V: $crate::traits::State<T>,
            D: $crate::traits::CallBackData,
        > $crate::interpolate::Interpolation<T, V> for $name<T, V, D> {
            fn interpolate(&mut self, t_interp: T) -> Result<V, $crate::Error<T, V>> {
                if self.t == self.t_prev {
                    if t_interp == self.t_prev {
                        return Ok(self.y_prev);
                    } else {
                        return Err($crate::Error::OutOfBounds { t_interp, t_prev: self.t_prev, t_curr: self.t });
                    }
                }
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::Error::OutOfBounds {
                        t_interp,
                        t_prev: self.t_prev,
                        t_curr: self.t });
                }

                let y_interp = $crate::interpolate::cubic_hermite_interpolate(
                    self.t_prev, self.t,
                    &self.y_prev, &self.y,
                    &self.dydt_prev, &self.dydt,
                    t_interp
                );

                Ok(y_interp)
            }
        }

// --- Builder Pattern Methods ---
        impl<
            T: $crate::traits::Real,
            V: $crate::traits::State<T>,
            D: $crate::traits::CallBackData,
        > $name<T, V, D> {
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
    };
}

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
        [0.5, 0.5],
        [0.5 + SQRT3 / 2.0, 0.5 - SQRT3 / 2.0]
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
        [5.0/18.0, 4.0/9.0, 5.0/18.0],
        [-5.0/6.0, 8.0/3.0, -5.0/6.0]
    ],
    c: [0.5 - SQRT15/10.0, 0.5, 0.5 + SQRT15/10.0],
    order: 6,
    stages: 3
);
