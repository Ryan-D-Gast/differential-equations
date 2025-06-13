//! Runge-Kutta solvers with support for dense output, embedded error estimation, and fixed steps.

use crate::{
    Error, Status,
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{ODENumericalMethod, ODE, methods::h_init},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
    tableau::ButcherTableau,
};

/// Runge-Kutta solver that can handle:
/// - Fixed-step methods with cubic Hermite interpolation
/// - Adaptive step methods with embedded error estimation and cubic Hermite interpolation
/// - Advanced methods with dense output interpolation using Butcher tableau coefficients
///
/// # Type Parameters
///
/// * `T`: Real number type (f32, f64)
/// * `V`: State vector type
/// * `D`: Callback data type
/// * `const S`: Number of stages in the method
/// * `const I`: Total number of stages including interpolation (equal to S for methods without dense output)
pub struct ExplicitRungeKutta<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> {
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: V,
    dydt: V,

    // Previous State
    h_prev: T,
    t_prev: T,
    y_prev: V,
    dydt_prev: V, // Added to support cubic Hermite interpolation

    // Stage values
    k: [V; I],

    // Constants from Butcher tableau
    c: [T; I],
    a: [[T; I]; I],
    b: [T; S],
    bh: Option<[T; S]>,  // Optional for methods without error estimation

    // Interpolation coefficients
    bi: Option<[[T; I]; I]>,  // Optional for methods without dense output
    cont: [T; I],

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
    steps: usize,

    // Status
    status: Status<T, V, D>,
    
    // Method info
    order: usize,
    stages: usize,
    dense_stages: usize,
}

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> Default for ExplicitRungeKutta<T, V, D, S, I> {
    fn default() -> Self {
        Self {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: V::zeros(),
            dydt: V::zeros(),
            h_prev: T::zero(),
            t_prev: T::zero(),
            y_prev: V::zeros(),
            dydt_prev: V::zeros(),
            k: [V::zeros(); I],
            c: [T::zero(); I],
            a: [[T::zero(); I]; I],
            b: [T::zero(); S],
            bh: None,
            bi: None,
            cont: [T::zero(); I],
            rtol: T::from_f64(1.0e-6).unwrap(),
            atol: T::from_f64(1.0e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 10000,
            max_rejects: 100,
            safety_factor: T::from_f64(0.9).unwrap(),
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(10.0).unwrap(),
            reject: false,
            n_stiff: 0,
            steps: 0,
            status: Status::Uninitialized,
            order: 0,
            stages: S,
            dense_stages: I,
        }
    }
}

// Fixed step methods (S = I, no embedded error estimation, cubic Hermite interpolation)
impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 1, 1> {
    /// Creates an Explicit Euler method (1st order, 1 stage).
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::euler`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::euler`].
    pub fn euler(h0: T) -> Self {
        let order = 1;
        let tableau = ButcherTableau::euler();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 2, 2> {
    /// Creates an Explicit Midpoint method (2nd order, 2 stages).
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::midpoint`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::midpoint`].
    pub fn midpoint(h0: T) -> Self {
        let order = 2;
        let tableau = ButcherTableau::midpoint();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }

    /// Creates an Explicit Heun method (2nd order, 2 stages).
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::heun`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::heun`].
    pub fn heun(h0: T) -> Self {
        let order = 2;
        let tableau = ButcherTableau::heun();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }

    /// Creates an Explicit Ralston method (2nd order, 2 stages).
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::ralston`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::ralston`].
    pub fn ralston(h0: T) -> Self {
        let order = 2;
        let tableau = ButcherTableau::ralston();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 4, 4> {
    /// Creates the classical 4th order Runge-Kutta method.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rk4`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rk4`].
    pub fn rk4(h0: T) -> Self {
        let order = 4;
        let tableau = ButcherTableau::rk4();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }
    
    /// Creates the three-eighths rule 4th order Runge-Kutta method.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::three_eighths`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::three_eighths`].
    pub fn three_eighths(h0: T) -> Self {
        let order = 4;
        let tableau = ButcherTableau::three_eighths();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;

        ExplicitRungeKutta {
            h0,
            c,
            a,
            b,
            order,
            ..Default::default()
        }
    }
}

// Adaptive step methods (embedded error estimation, cubic Hermite interpolation)
impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 6, 6> {
    /// Creates a Runge-Kutta-Fehlberg 4(5) method with error estimation.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkf45`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkf45`].
    pub fn rkf45() -> Self {
        let order = 5;
        let tableau = ButcherTableau::rkf45();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh.unwrap();

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh: Some(bh),
            order,
            ..Default::default()
        }
    }
    
    /// Creates a Cash-Karp 4(5) method with error estimation.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::cash_karp`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::cash_karp`].
    pub fn cash_karp() -> Self {
        let order = 5;
        let tableau = ButcherTableau::cash_karp();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh.unwrap();

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh: Some(bh),
            order,
            ..Default::default()
        }
    }
}

// Methods with dense output (Verner's methods, already implemented)
impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 10, 13> {
    /// Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 6th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv766e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv766e`].
    pub fn rkv766e() -> Self {
        let order = 7;
        let tableau = ButcherTableau::rkv766e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 10, 16> {
    /// Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 7th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv767e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv767e`].
    pub fn rkv767e() -> Self {
        let order = 7;
        let tableau = ButcherTableau::<T, 10, 16>::rkv767e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 13, 17> {
    /// Creates a ExplicitRungeKutta 8(7) method with 13 stages with 7th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv877e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv877e`].
    pub fn rkv877e() -> Self {
        let order = 8;
        let tableau = ButcherTableau::rkv877e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 13, 21> {
    /// Creates a ExplicitRungeKutta 8(7) method with 13 stages with 8th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv878e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv878e`].
    pub fn rkv878e() -> Self {
        let order = 8;
        let tableau = ButcherTableau::rkv878e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 16, 21> {
    /// Creates a ExplicitRungeKutta 9(8) method with 16 stages with 8th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv878e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv878e`].
    pub fn rkv988e() -> Self {
        let order = 9;
        let tableau = ButcherTableau::rkv988e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<T, V, D, 16, 26> {
    /// Creates a ExplicitRungeKutta 9(8) method with 16 stages with 9th order interpolant.
    ///
    /// Uses the Butcher tableau from [`ButcherTableau::rkv878e`].
    /// 
    /// For detailed coefficients and method properties, see [`ButcherTableau::rkv878e`].
    pub fn rkv989e() -> Self {
        let order = 9;
        let tableau = ButcherTableau::rkv989e();
        let c = tableau.c;
        let a = tableau.a;
        let b = tableau.b;
        let bh = tableau.bh;
        let bi = tableau.bi;

        ExplicitRungeKutta {
            c,
            a,
            b,
            bh,
            bi,
            order,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> ODENumericalMethod<T, V, D> for ExplicitRungeKutta<T, V, D, S, I> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Only use adaptive step size calculation if the method supports it
            if self.bh.is_some() {
                self.h0 = h_init(ode, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max);
                evals.fcn += 2;
            } else {
                // Simple default step size for fixed-step methods
                let duration = (tf - t0).abs();
                let default_steps = T::from_usize(100).unwrap();
                self.h0 = duration / default_steps;
            }
        }

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.reject = false;
        self.n_stiff = 0;

        // Initialize State
        self.t = t0;
        self.y = y0.clone();
        ode.diff(t0, y0, &mut self.dydt);
        evals.fcn += 1;

        // Initialize previous state
        self.t_prev = t0;
        self.y_prev = y0.clone();
        self.dydt_prev = self.dydt.clone();

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check step size
        if self.h.abs() < T::default_epsilon() {
            self.status = Status::Error(Error::StepSize {
                t: self.t, y: self.y
            });
            return Err(Error::StepSize {
                t: self.t, y: self.y
            });
        }

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

        // Save k[0] as the current derivative
        self.k[0] = self.dydt.clone();

        // Compute stages
        for i in 1..self.stages {
            let mut y_stage = self.y.clone();

            for j in 0..i {
                y_stage += self.k[j] * (self.a[i][j] * self.h);
            }

            ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
        }
        evals.fcn += self.stages - 1; // We already have k[0]

        // Store current state before update for interpolation
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.k[0];

        // For methods without embedded error estimation, simply update the solution
        if self.bh.is_none() {
            // Compute solution
            let mut y_next = self.y;
            for i in 0..self.stages {
                y_next += self.k[i] * (self.b[i] * self.h);
            }

            // Update state
            self.t += self.h;
            self.y = y_next;
            
            // Calculate new derivative for next step
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;
            
            self.status = Status::Solving;
            return Ok(evals);
        }
        
        // For adaptive methods with error estimation
        // Compute higher order solution
        let mut y_high = self.y;
        for i in 0..self.stages {
            y_high += self.k[i] * (self.b[i] * self.h);
        }

        // Compute lower order solution for error estimation
        let mut y_low = self.y;
        if let Some(bh) = &self.bh {
            for i in 0..self.stages {
                y_low += self.k[i] * (bh[i] * self.h);
            }
        }

        // Compute error estimate
        let err = y_high - y_low;

        // Calculate error norm
        let mut err_norm: T = T::zero();

        // Iterate through state elements
        for n in 0..self.y.len() {
            let tol = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
            err_norm = err_norm.max((err.get(n) / tol).abs());
        };

        // Determine if step is accepted
        if err_norm <= T::one() {
            // Log previous state
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.h_prev = self.h;

            if self.reject {
                self.n_stiff = 0;
                self.reject = false;
                self.status = Status::Solving;
            }

            // If method has dense output stages, compute them
            if self.bi.is_some() {
                // Compute extra stages for dense output
                for i in 0..(I - S) {
                    let mut y_stage = self.y;
                    for j in 0..self.stages + i {
                        y_stage += self.k[j] * (self.a[self.stages + i][j] * self.h);
                    }

                    ode.diff(self.t + self.c[self.stages + i] * self.h, &y_stage, &mut self.k[self.stages + i]);
                }
                evals.fcn += I - S;
            }

            // Update state with the higher-order solution
            self.t += self.h;
            self.y = y_high;
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;
        } else {
            // Step rejected
            self.reject = true;
            self.status = Status::RejectedStep;
            self.n_stiff += 1;

            // Check for stiffness
            if self.n_stiff >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t, y: self.y
                });
                return Err(Error::Stiffness {
                    t: self.t, y: self.y
                });
            }
        }

        // Calculate new step size for adaptive methods
        let order = T::from_usize(self.order).unwrap();
        let err_order = T::one() / order;

        // Step size controller
        let scale = self.safety_factor * err_norm.powf(-err_order);
        let scale = scale.max(self.min_scale).min(self.max_scale);
        self.h *= scale;

        // Ensure step size is within bounds
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);
        
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

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<T, V, D, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t
            });
        }

        // If method has dense output coefficients, use them
        if self.bi.is_some() {
            // Calculate the normalized distance within the step [0, 1]
            let s = (t_interp - self.t_prev) / self.h_prev;
            
            // Get the interpolation coefficients
            let bi = self.bi.as_ref().unwrap();

            // Compute the interpolation coefficients using Horner's method
            for i in 0..self.dense_stages {
                // Start with the highest-order term
                self.cont[i] = bi[i][self.dense_stages - 1];

                // Apply Horner's method
                for j in (0..self.dense_stages - 1).rev() {
                    self.cont[i] = self.cont[i] * s + bi[i][j];
                }

                // Multiply by s
                self.cont[i] *= s;
            }

            // Compute the interpolated value
            let mut y_interp = self.y_prev;
            for i in 0..I {
                if i < self.k.len() && i < self.cont.len() {
                    y_interp += self.k[i] * (self.cont[i] * self.h_prev);
                }
            }

            return Ok(y_interp);
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

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> ExplicitRungeKutta<T, V, D, S, I> {
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

    /// Set the initial step size
    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
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
        self.order
    }

    /// Get the number of stages in the method
    pub fn stages(&self) -> usize {
        self.stages
    }

    /// Get the number of terms in the dense output interpolation polynomial
    pub fn dense_stages(&self) -> usize {
        self.dense_stages
    }
}