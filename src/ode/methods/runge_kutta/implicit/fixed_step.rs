//! Fixed-step implicit Runge-Kutta methods for solving ordinary differential equations.

/// Macro to create a fixed-step implicit Runge-Kutta solver from a Butcher tableau.
///
/// This macro generates the necessary struct and trait implementations for a fixed-step
/// implicit Runge-Kutta method. It uses a simple fixed-point iteration to solve the
/// implicit stage equations.
///
/// # Arguments
///
/// * `name`: Name of the solver struct to create
/// * `a`: Matrix of coefficients for intermediate stages (can be non-zero on diagonal/upper triangle)
/// * `b`: Weights for final summation
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the method
/// * `stages`: Number of stages in the method
///
/// # Note on Solver
/// The implicit stage equations `k_i = f(t_n + c_i*h, y_n + h * sum(a_{ij}*k_j))` are solved
/// using fixed-point iteration. This is simple but may fail to converge for stiff problems
/// unless `h` is sufficiently small (`h * L < 1`, where `L` is the Lipschitz constant).
/// More robust solvers (like Newton's method) require Jacobians and linear algebra.
///
/// # Example
/// ```
/// use differential_equations::implicit_runge_kutta_method;
///
/// // Define Implicit Euler method
/// implicit_runge_kutta_method!(
///     /// Implicit Euler (Backward Euler) Method (1st Order)
///     name: ImplicitEulerExample,
///     a: [[1.0]],
///     b: [1.0],
///     c: [1.0],
///     order: 1,
///     stages: 1
/// );
/// ```
#[macro_export]
macro_rules! implicit_runge_kutta_method {
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
        #[doc = "This fixed-step implicit solver was automatically generated using the `implicit_runge_kutta_method` macro."]
        #[doc = " It uses fixed-point iteration to solve the stage equations."]
        pub struct $name<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> {
            // Step Size
            pub h: T,

            // Current State
            t: T,
            y: V,

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
            b: [T; $stages],
            c: [T; $stages],

            // --- Solver Settings ---
            pub max_iter: usize, // Max iterations for fixed-point solver
            pub tol: T,          // Tolerance for fixed-point solver convergence

            // Status & Counters
            status: $crate::Status<T, V, D>,
            steps: usize,
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> Default for $name<T, V, D> {
            fn default() -> Self {
                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let b_t: [T; $stages] = $b.map(|x| T::from_f64(x).unwrap());
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                $name {
                    h: T::from_f64(0.01).unwrap(), // Default fixed step size
                    t: T::zero(),
                    y: V::zeros(),
                    t_prev: T::zero(),
                    y_prev: V::zeros(),
                    dydt_prev: V::zeros(),
                    k: [V::zeros(); $stages],
                    y_stage: [V::zeros(); $stages],
                    k_new: [V::zeros(); $stages],
                    a: a_t,
                    b: b_t,
                    c: c_t,
                    max_iter: 50, // Default max iterations
                    tol: T::from_f64(1e-8).unwrap(), // Default tolerance
                    status: $crate::Status::Uninitialized,
                    steps: 0,
                }
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::ode::NumericalMethod<T, V, D> for $name<T, V, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>
            {
                let mut evals = $crate::alias::Evals::new();

                 if self.h == T::zero() {
                    return Err($crate::Error::BadInput {
                        msg: concat!(stringify!($name), " requires a non-zero fixed step size 'h' to be set.").to_string(),
                    });
                }
                // Basic validation
                self.h = $crate::utils::validate_step_size_parameters::<T, V, D>(self.h, T::zero(), T::infinity(), t0, tf)?;

                // Initialize State
                self.t = t0;
                self.y = *y0;
                self.t_prev = t0;
                self.y_prev = *y0;

                // Calculate initial derivative f(t0, y0) for interpolation
                ode.diff(t0, y0, &mut self.dydt_prev);
                evals.fcn += 1;

                // Reset counters
                self.steps = 0;

                self.status = $crate::Status::Initialized;
                Ok(evals)
            }

            fn step<F>(&mut self, ode: &F) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>
            {
                let mut evals = $crate::alias::Evals::new();

                // --- Fixed-Point Iteration for stage derivatives k_i ---
                // Initial guess: k_i^{(0)} = f(t_n, y_n) (stored in self.dydt_prev)
                for i in 0..$stages {
                    self.k[i] = self.dydt_prev;
                }

                let mut converged = false;
                for _iter in 0..self.max_iter {
                    let mut max_diff_sq = T::zero();

                    // Calculate next iteration k_i^{(m+1)} based on k_j^{(m)}
                    for i in 0..$stages {
                        // Calculate stage value y_stage = y_n + h * sum(a_ij * k_j^{(m)})
                        self.y_stage[i] = self.y;
                        for j in 0..$stages {
                            // Use current k values from this iteration
                            self.y_stage[i] += self.k[j] * (self.a[i][j] * self.h);
                        }

                        // Evaluate f at stage time and value: f(t_n + c_i*h, y_stage)
                        ode.diff(self.t + self.c[i] * self.h, &self.y_stage[i], &mut self.k_new[i]);
                        evals.fcn += 1;
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
                    self.status = $crate::Status::Error($crate::Error::StepSize { t: self.t, y: self.y });
                    return Err($crate::Error::StepSize { t: self.t, y: self.y });
                }

                // --- Iteration converged, compute final update ---
                self.steps += 1;

                // Store previous state
                self.t_prev = self.t;
                self.y_prev = self.y;
                // Note: self.dydt_prev remains f(t_prev, y_prev)

                // Compute the final update y_{n+1} = y_n + h * sum(b_i * k_i)
                let mut delta_y = V::zeros();
                for i in 0..$stages {
                    delta_y += self.k[i] * (self.b[i] * self.h);
                }

                // Update state
                self.y += delta_y;
                self.t += self.h;

                // Calculate derivative at the new point for the *next* step's prediction
                // and for interpolation purposes.
                ode.diff(self.t, &self.y, &mut self.dydt_prev); // Store f(t_new, y_new) in dydt_prev for next step
                evals.fcn += 1; // Count this evaluation

                self.status = $crate::Status::Solving;
                Ok(evals) // Return evals for this step
            }

            // --- Standard trait methods ---
            fn t(&self) -> T { self.t }
            fn y(&self) -> &V { &self.y }
            fn t_prev(&self) -> T { self.t_prev }
            fn y_prev(&self) -> &V { &self.y_prev }
            fn h(&self) -> T { self.h }
            fn set_h(&mut self, h: T) { self.h = h; }
            fn status(&self) -> &$crate::Status<T, V, D> { &self.status }
            fn set_status(&mut self, status: $crate::Status<T, V, D>) { self.status = status; }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::interpolate::Interpolation<T, V> for $name<T, V, D> {
            fn interpolate(&mut self, t_interp: T) -> Result<V, $crate::Error<T, V>> {
                 if self.t == self.t_prev { // Handle case before first step
                     if t_interp == self.t_prev {
                         return Ok(self.y_prev);
                     } else {
                         return Err($crate::Error::OutOfBounds { t_interp, t_prev: self.t_prev, t_curr: self.t });
                     }
                 }

                // Check if t is within the bounds of the current step
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::Error::OutOfBounds {
                        t_interp,
                        t_prev: self.t_prev,
                        t_curr: self.t });
                }

                // Use cubic Hermite interpolation between (t_prev, y_prev, dydt_prev) and (t, y, k[0])
                let y_interp = $crate::interpolate::cubic_hermite_interpolate(
                    self.t_prev, self.t,
                    &self.y_prev, &self.y,
                    &self.dydt_prev, &self.k[0],
                    t_interp
                );

                Ok(y_interp)
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $name<T, V, D> {
            /// Create a new solver instance with default settings.
            pub fn new(h: T) -> Self {
                $name {
                    h,
                    ..Default::default()
                }
            }

            /// Set the fixed step size `h`.
            pub fn h(mut self, h: T) -> Self {
                self.h = h;
                self
            }

            /// Set the maximum number of fixed-point iterations per step.
            pub fn max_iter(mut self, iter: usize) -> Self {
                self.max_iter = iter;
                self
            }

            /// Set the tolerance for fixed-point iteration convergence.
            pub fn tol(mut self, tol: T) -> Self {
                self.tol = tol;
                self
            }
        }
    };
}

implicit_runge_kutta_method!(
    /// Implicit Euler (Backward Euler) Method (1st Order)
    ///
    /// Solves `y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})`.
    /// The Butcher Tableau is:
    /// ```text
    /// 1 | 1
    /// -----
    ///   | 1
    /// ```
    name: BackwardEuler,
    a: [[1.0]],
    b: [1.0],
    c: [1.0],
    order: 1,
    stages: 1
);

implicit_runge_kutta_method!(
    /// Crank-Nicolson Method (Trapezoidal Rule) (2nd Order)
    ///
    /// Solves `y_{n+1} = y_n + 0.5*h * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))`.
    /// This is often implemented as a 2-stage implicit method.
    /// Stage 1: `k1 = f(t_n, y_n)` (explicit)
    /// Stage 2: `k2 = f(t_{n+1}, y_n + 0.5*h*k1 + 0.5*h*k2)` (implicit)
    /// Update: `y_{n+1} = y_n + 0.5*h*k1 + 0.5*h*k2`
    /// The Butcher Tableau is:
    /// ```text
    /// 0   | 0   0
    /// 1   | 1/2 1/2
    /// --------------
    ///     | 1/2 1/2
    /// ```
    /// Note: The fixed-point solver in this macro solves for *all* stages simultaneously.
    /// For Crank-Nicolson, k1 is explicit, but the solver treats it implicitly.
    /// This works but is less efficient than a specialized implementation.
    name: CrankNicolson,
    a: [[0.0, 0.0],
        [0.5, 0.5]],
    b: [0.5, 0.5],
    c: [0.0, 1.0],
    order: 2,
    stages: 2
);
