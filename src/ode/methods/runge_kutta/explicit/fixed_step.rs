//! Fixed-step Runge-Kutta methods for solving ordinary differential equations.

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
/// use differential_equations::runge_kutta_method;
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
        pub struct $name<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> {
            // Step Size
            pub h: T,

            // Current State
            t: T,
            y: V,

            // Previous State
            t_prev: T,
            y_prev: V,
            dydt_prev: V,

            // Stage values (fixed size arrays of Vectors)
            k: [V; $stages],

            // Constants from Butcher tableau (fixed size arrays)
            a: [[T; $stages]; $stages],
            b: [T; $stages],
            c: [T; $stages],

            // Status
            status: $crate::Status<T, V, D>,
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> Default for $name<T, V, D> {
            fn default() -> Self {
                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let b_t: [T; $stages] = $b.map(|x| T::from_f64(x).unwrap());
                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                $name {
                    h: T::from_f64(0.1).unwrap(),
                    t: T::from_f64(0.0).unwrap(),
                    y: V::zeros(),
                    t_prev: T::from_f64(0.0).unwrap(),
                    y_prev: V::zeros(),
                    dydt_prev: V::zeros(),
                    // Initialize k vectors with zeros
                    k: [V::zeros(); $stages],
                    // Use the converted Butcher tableau
                    a: a_t,
                    b: b_t,
                    c: c_t,
                    status: $crate::Status::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::ode::ODENumericalMethod<T, V, D> for $name<T, V, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &V) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>
            {
                let mut evals = $crate::alias::Evals::new();

                // Check Bounds
                match $crate::utils::validate_step_size_parameters::<T, V, D>(self.h, T::zero(), T::infinity(), t0, tf) {
                    Ok(_) => {},
                    Err(e) => return Err(e),
                }

                // Initialize State
                self.t = t0;
                self.y = y.clone();
                ode.diff(t0, y, &mut self.k[0]);
                evals.fcn += 1;

                // Initialize previous state
                self.t_prev = t0;
                self.y_prev = y.clone();
                self.dydt_prev = self.k[0];

                // Initialize Status
                self.status = $crate::Status::Initialized;

                Ok(evals)
            }

            fn step<F>(&mut self, ode: &F) -> Result<$crate::alias::Evals, $crate::Error<T, V>>
            where
                F: $crate::ode::ODE<T, V, D>
            {
                let mut evals = $crate::alias::Evals::new();

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
                evals.fcn += $stages;

                // Compute the final update
                let mut delta_y = V::zeros();
                for i in 0..$stages {
                    delta_y += self.k[i] * (self.b[i] * self.h);
                }

                // Update state
                self.y += delta_y;
                self.t += self.h;

                Ok(evals)
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &V {
                &self.y
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &V {
                &self.y_prev
            }

            fn h(&self) -> T {
                self.h
            }

            fn set_h(&mut self, h: T) {
                self.h = h;
            }

            fn status(&self) -> &$crate::Status<T, V, D> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::Status<T, V, D>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $crate::interpolate::Interpolation<T, V> for $name<T, V, D> {
            fn interpolate(&mut self, t_interp: T) -> Result<V, $crate::Error<T, V>> {
                // Check if t is within the bounds of the current step
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::Error::OutOfBounds {
                        t_interp,
                        t_prev: self.t_prev,
                        t_curr: self.t });
                }

                let y_interp = $crate::interpolate::cubic_hermite_interpolate(self.t_prev, self.t, &self.y_prev, &self.y, &self.dydt_prev, &self.k[0], t_interp);

                Ok(y_interp)
            }
        }

        impl<T: $crate::traits::Real, V: $crate::traits::State<T>, D: $crate::traits::CallBackData> $name<T, V, D> {
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

runge_kutta_method!(
    /// Euler's Method (1st Order Runge-Kutta) for solving ordinary differential equations.
    ///
    /// Euler's method is the simplest form of Runge-Kutta methods, and is a first-order method also known as RK1.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0 | 0
    /// -----
    ///   | 1
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Euler_method)
    name: Euler,
    a: [[0.0]],
    b: [1.0],
    c: [0.0],
    order: 1,
    stages: 1
);

runge_kutta_method!(
    /// Midpoint Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1/2 | 1/2
    /// ------------
    ///     | 0   1
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Midpoint_method)
    name: Midpoint,
    a: [[0.0, 0.0],
        [0.5, 0.0]],
    b: [0.0, 1.0],
    c: [0.0, 0.5],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Heun's Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1   | 1
    /// ------------
    ///     | 1/2 1/2
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Heun%27s_method)
    name: Heun,
    a: [[0.0, 0.0],
        [1.0, 0.0]],
    b: [0.5, 0.5],
    c: [0.0, 1.0],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Ralston's Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 2/3 | 2/3
    /// ------------
    ///     | 1/4 3/4
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Second-order_methods_with_two_stages)
    name: Ralston,
    a: [[0.0, 0.0],
        [2.0/3.0, 0.0]],
    b: [1.0/4.0, 3.0/4.0],
    c: [0.0, 2.0/3.0],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Classic Runge-Kutta 4 method for solving ordinary differential equations.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 0.5 | 0.5
    /// 0.5 | 0   0.5
    /// 1   | 0   0   1
    /// ---------------------
    ///    | 1/6 1/3 1/3 1/6
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples)
    name: RK4,
    a: [[0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]],
    b: [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
    c: [0.0, 0.5, 0.5, 1.0],
    order: 4,
    stages: 4
);

runge_kutta_method!(
    /// Three-Eighths Rule (4th Order Runge-Kutta) for solving ordinary differential equations.
    /// The primary advantage this method has is that almost all of the error coefficients
    /// are smaller than in the popular method, but it requires slightly more FLOPs
    /// (floating-point operations) per time step.
    ///
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1/3 | 1/3
    /// 2/3 | -1/3 1
    /// 1   | 1   -1   1
    /// ---------------------
    ///   | 1/8 3/8 3/8 1/8
    /// ```
    ///
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples)
    ///
    name: ThreeEights,
    a: [[0.0, 0.0, 0.0, 0.0],
        [1.0/3.0, 0.0, 0.0, 0.0],
        [-1.0/3.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0, 0.0]],
    b: [1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0],
    c: [0.0, 1.0/3.0, 2.0/3.0, 1.0],
    order: 4,
    stages: 4
);
