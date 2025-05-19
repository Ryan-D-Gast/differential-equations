//! DOP853 NumericalMethod for Ordinary Differential Equations.

use crate::{
    Error, Status,
    alias::Evals,
    interpolate::Interpolation,
    ode::{NumericalMethod, ODE, methods::h_init},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

/// Dormand Prince 8(5, 3) Method for solving ordinary differential equations.
/// 8th order Dormand Prince method with embedded 5th order error estimation and 3rd order interpolation.
/// The resulting interpolant is of order 7.
///
/// Builds should begin with weight, normal, dense, or even methods.
/// and then chain the other methods to set the parameters.
/// The defaults should be great for most cases.
///
/// # Example
/// ```
/// use differential_equations::prelude::*;
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
/// impl ODE<f64, SVector<f64, 2>> for Example {
///    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///       dydt[0] = y[1];
///       dydt[1] = -y[0];
///   }
/// }
/// let solution = ODEProblem::new(Example, t0, tf, y0).solve(&mut dop853).unwrap();
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
pub struct DOP853<T: Real, V: State<T>, D: CallBackData> {
    // Initial Conditions
    pub h0: T, // Initial Step Size

    // Current iteration
    t: T,
    y: V,
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
    status: Status<T, V, D>,
    steps: usize,      // Number of Steps
    n_accepted: usize, // Number of Accepted Steps

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
    b_dense: [[T; 16]; 4],

    // Derivatives - using array instead of individually numbered variables
    k: [V; 12], // k[0] is derivative at t, others are stage derivatives

    // For Interpolation - using array instead of individually numbered variables
    y_old: V,     // State at Previous Step
    t_old: T,     // Time of Previous Step
    h_old: T,     // Step Size of Previous Step
    cont: [V; 8], // Interpolation coefficients
}

impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for DOP853<T, V, D> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Set Current State as Initial State
        self.t = t0;
        self.y = *y0;

        // Calculate derivative at t0
        ode.diff(t0, y0, &mut self.k[0]);
        evals.fcn += 1; // Increment function evaluations for initial derivative calculation

        // Initialize Previous State
        self.t_old = self.t;
        self.y_old = self.y;

        // Calculate Initial Step
        if self.h0 == T::zero() {
            self.h0 = h_init(
                ode, t0, tf, y0, 8, self.rtol, self.atol, self.h_min, self.h_max,
            );
            evals.fcn += 1; // Increment function evaluations for initial step size calculation

            // Adjust h0 to be within bounds
            let posneg = (tf - t0).signum();
            if self.h0.abs() < self.h_min.abs() {
                self.h0 = self.h_min.abs() * posneg;
            } else if self.h0.abs() > self.h_max.abs() {
                self.h0 = self.h_max.abs() * posneg;
            }
        }

        // Check if h0 is within bounds, and h_min and h_max are valid
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Make sure iteration variables are reset
        self.h_lamb = T::zero();
        self.non_stiff_counter = 0;
        self.stiffness_counter = 0;

        // NumericalMethod is ready to go
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check if Max Steps Reached
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
        }

        // Check if Step Size is too smaller then machine default_epsilon
        if self.h.abs() < T::default_epsilon() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y,
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y,
            });
        }

        // The twelve stages
        ode.diff(
            self.t + self.c[1] * self.h,
            &(self.y + self.k[0] * (self.a[1][0] * self.h)),
            &mut self.k[1],
        );
        ode.diff(
            self.t + self.c[2] * self.h,
            &(self.y + self.k[0] * (self.a[2][0] * self.h) + self.k[1] * (self.a[2][1] * self.h)),
            &mut self.k[2],
        );
        ode.diff(
            self.t + self.c[3] * self.h,
            &(self.y + self.k[0] * (self.a[3][0] * self.h) + self.k[2] * (self.a[3][2] * self.h)),
            &mut self.k[3],
        );
        ode.diff(
            self.t + self.c[4] * self.h,
            &(self.y
                + self.k[0] * (self.a[4][0] * self.h)
                + self.k[2] * (self.a[4][2] * self.h)
                + self.k[3] * (self.a[4][3] * self.h)),
            &mut self.k[4],
        );
        ode.diff(
            self.t + self.c[5] * self.h,
            &(self.y
                + self.k[0] * (self.a[5][0] * self.h)
                + self.k[3] * (self.a[5][3] * self.h)
                + self.k[4] * (self.a[5][4] * self.h)),
            &mut self.k[5],
        );
        ode.diff(
            self.t + self.c[6] * self.h,
            &(self.y
                + self.k[0] * (self.a[6][0] * self.h)
                + self.k[3] * (self.a[6][3] * self.h)
                + self.k[4] * (self.a[6][4] * self.h)
                + self.k[5] * (self.a[6][5] * self.h)),
            &mut self.k[6],
        );
        ode.diff(
            self.t + self.c[7] * self.h,
            &(self.y
                + self.k[0] * (self.a[7][0] * self.h)
                + self.k[3] * (self.a[7][3] * self.h)
                + self.k[4] * (self.a[7][4] * self.h)
                + self.k[5] * (self.a[7][5] * self.h)
                + self.k[6] * (self.a[7][6] * self.h)),
            &mut self.k[7],
        );
        ode.diff(
            self.t + self.c[8] * self.h,
            &(self.y
                + self.k[0] * (self.a[8][0] * self.h)
                + self.k[3] * (self.a[8][3] * self.h)
                + self.k[4] * (self.a[8][4] * self.h)
                + self.k[5] * (self.a[8][5] * self.h)
                + self.k[6] * (self.a[8][6] * self.h)
                + self.k[7] * (self.a[8][7] * self.h)),
            &mut self.k[8],
        );
        ode.diff(
            self.t + self.c[9] * self.h,
            &(self.y
                + self.k[0] * (self.a[9][0] * self.h)
                + self.k[3] * (self.a[9][3] * self.h)
                + self.k[4] * (self.a[9][4] * self.h)
                + self.k[5] * (self.a[9][5] * self.h)
                + self.k[6] * (self.a[9][6] * self.h)
                + self.k[7] * (self.a[9][7] * self.h)
                + self.k[8] * (self.a[9][8] * self.h)),
            &mut self.k[9],
        );
        ode.diff(
            self.t + self.c[10] * self.h,
            &(self.y
                + self.k[0] * (self.a[10][0] * self.h)
                + self.k[3] * (self.a[10][3] * self.h)
                + self.k[4] * (self.a[10][4] * self.h)
                + self.k[5] * (self.a[10][5] * self.h)
                + self.k[6] * (self.a[10][6] * self.h)
                + self.k[7] * (self.a[10][7] * self.h)
                + self.k[8] * (self.a[10][8] * self.h)
                + self.k[9] * (self.a[10][9] * self.h)),
            &mut self.k[1],
        );
        let t_new = self.t + self.h;
        let yy1 = self.y
            + self.k[0] * (self.a[11][0] * self.h)
            + self.k[3] * (self.a[11][3] * self.h)
            + self.k[4] * (self.a[11][4] * self.h)
            + self.k[5] * (self.a[11][5] * self.h)
            + self.k[6] * (self.a[11][6] * self.h)
            + self.k[7] * (self.a[11][7] * self.h)
            + self.k[8] * (self.a[11][8] * self.h)
            + self.k[9] * (self.a[11][9] * self.h)
            + self.k[1] * (self.a[11][10] * self.h);
        ode.diff(t_new, &yy1, &mut self.k[2]);
        self.k[3] = self.k[0] * self.b[0]
            + self.k[5] * self.b[5]
            + self.k[6] * self.b[6]
            + self.k[7] * self.b[7]
            + self.k[8] * self.b[8]
            + self.k[9] * self.b[9]
            + self.k[1] * self.b[10]
            + self.k[2] * self.b[11];
        self.k[4] = self.y + self.k[3] * self.h;

        // Log evals
        evals.fcn += 11; // Increment function evaluations for the 11 derivative calculations

        // Error Estimation
        let mut err = T::zero();
        let mut err2 = T::zero();

        let n = self.y.len();
        for i in 0..n {
            let sk = self.atol + self.rtol * self.y.get(i).abs().max(self.k[4].get(i).abs());
            let erri = self.k[3].get(i)
                - self.bhh[0] * self.k[0].get(i)
                - self.bhh[1] * self.k[8].get(i)
                - self.bhh[2] * self.k[2].get(i);
            err2 += (erri / sk).powi(2);
            let erri = self.er[0] * self.k[0].get(i)
                + self.er[5] * self.k[5].get(i)
                + self.er[6] * self.k[6].get(i)
                + self.er[7] * self.k[7].get(i)
                + self.er[8] * self.k[8].get(i)
                + self.er[9] * self.k[9].get(i)
                + self.er[10] * self.k[1].get(i)
                + self.er[11] * self.k[2].get(i);
            err += (erri / sk).powi(2);
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
            let y_new = self.k[4];
            ode.diff(t_new, &y_new, &mut self.k[3]);
            evals.fcn += 1; // Increment function evaluations for the new derivative calculation

            // stiffness detection
            if self.n_accepted % self.n_stiff == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = self.k[3] - self.k[2];
                for i in 0..sqr.len() {
                    stdnum += sqr.get(i).powi(2);
                }
                let sqr = self.k[4] - yy1;
                for i in 0..sqr.len() {
                    stden += sqr.get(i).powi(2);
                }

                if stden > T::zero() {
                    self.h_lamb = self.h * (stdnum / stden).sqrt();
                }
                if self.h_lamb > T::from_f64(6.1).unwrap() {
                    self.non_stiff_counter = 0;
                    self.stiffness_counter += 1;
                    if self.stiffness_counter == 15 {
                        // Early Exit Stiffness Detected
                        self.status = Status::Error(Error::Stiffness {
                            t: self.t,
                            y: self.y,
                        });
                        return Err(Error::Stiffness {
                            t: self.t,
                            y: self.y,
                        });
                    }
                } else {
                    self.non_stiff_counter += 1;
                    if self.non_stiff_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }

            // Preperation for dense / continuous output
            self.cont[0] = self.y;
            let ydiff = self.k[4] - self.y;
            self.cont[1] = ydiff;
            let bspl = self.k[0] * self.h - ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.k[3] * self.h - bspl;

            self.cont[4] = self.k[0] * self.b_dense[0][0]
                + self.k[5] * self.b_dense[0][5]
                + self.k[6] * self.b_dense[0][6]
                + self.k[7] * self.b_dense[0][7]
                + self.k[8] * self.b_dense[0][8]
                + self.k[9] * self.b_dense[0][9]
                + self.k[1] * self.b_dense[0][10]
                + self.k[2] * self.b_dense[0][11];

            self.cont[5] = self.k[0] * self.b_dense[1][0]
                + self.k[5] * self.b_dense[1][5]
                + self.k[6] * self.b_dense[1][6]
                + self.k[7] * self.b_dense[1][7]
                + self.k[8] * self.b_dense[1][8]
                + self.k[9] * self.b_dense[1][9]
                + self.k[1] * self.b_dense[1][10]
                + self.k[2] * self.b_dense[1][11];

            self.cont[6] = self.k[0] * self.b_dense[2][0]
                + self.k[5] * self.b_dense[2][5]
                + self.k[6] * self.b_dense[2][6]
                + self.k[7] * self.b_dense[2][7]
                + self.k[8] * self.b_dense[2][8]
                + self.k[9] * self.b_dense[2][9]
                + self.k[1] * self.b_dense[2][10]
                + self.k[2] * self.b_dense[2][11];

            self.cont[7] = self.k[0] * self.b_dense[3][0]
                + self.k[5] * self.b_dense[3][5]
                + self.k[6] * self.b_dense[3][6]
                + self.k[7] * self.b_dense[3][7]
                + self.k[8] * self.b_dense[3][8]
                + self.k[9] * self.b_dense[3][9]
                + self.k[1] * self.b_dense[3][10]
                + self.k[2] * self.b_dense[3][11];

            ode.diff(
                self.t + self.c_dense[0] * self.h,
                &(self.y
                    + (self.k[0] * self.a_dense[0][0]
                        + self.k[6] * self.a_dense[0][6]
                        + self.k[7] * self.a_dense[0][7]
                        + self.k[8] * self.a_dense[0][8]
                        + self.k[9] * self.a_dense[0][9]
                        + self.k[1] * self.a_dense[0][10]
                        + self.k[2] * self.a_dense[0][11]
                        + self.k[3] * self.a_dense[0][12])
                        * self.h),
                &mut self.k[9],
            );

            ode.diff(
                self.t + self.c_dense[1] * self.h,
                &(self.y
                    + (self.k[0] * self.a_dense[1][0]
                        + self.k[5] * self.a_dense[1][5]
                        + self.k[6] * self.a_dense[1][6]
                        + self.k[7] * self.a_dense[1][7]
                        + self.k[1] * self.a_dense[1][10]
                        + self.k[2] * self.a_dense[1][11]
                        + self.k[3] * self.a_dense[1][12]
                        + self.k[9] * self.a_dense[1][13])
                        * self.h),
                &mut self.k[1],
            );

            ode.diff(
                self.t + self.c_dense[2] * self.h,
                &(self.y
                    + (self.k[0] * self.a_dense[2][0]
                        + self.k[5] * self.a_dense[2][5]
                        + self.k[6] * self.a_dense[2][6]
                        + self.k[7] * self.a_dense[2][7]
                        + self.k[8] * self.a_dense[2][8]
                        + self.k[3] * self.a_dense[2][12]
                        + self.k[9] * self.a_dense[2][13]
                        + self.k[1] * self.a_dense[2][14])
                        * self.h),
                &mut self.k[2],
            );

            // Log evals
            evals.fcn += 3; // Increment function evaluations for the dense output calculations

            // Final preparation - add contributions from the extra stages and scale
            self.cont[4] = (self.cont[4]
                + self.k[3] * self.b_dense[0][12]
                + self.k[9] * self.b_dense[0][13]
                + self.k[1] * self.b_dense[0][14]
                + self.k[2] * self.b_dense[0][15])
                * self.h;

            self.cont[5] = (self.cont[5]
                + self.k[3] * self.b_dense[1][12]
                + self.k[9] * self.b_dense[1][13]
                + self.k[1] * self.b_dense[1][14]
                + self.k[2] * self.b_dense[1][15])
                * self.h;

            self.cont[6] = (self.cont[6]
                + self.k[3] * self.b_dense[2][12]
                + self.k[9] * self.b_dense[2][13]
                + self.k[1] * self.b_dense[2][14]
                + self.k[2] * self.b_dense[2][15])
                * self.h;

            self.cont[7] = (self.cont[7]
                + self.k[3] * self.b_dense[3][12]
                + self.k[9] * self.b_dense[3][13]
                + self.k[1] * self.b_dense[3][14]
                + self.k[2] * self.b_dense[3][15])
                * self.h;

            // For Interpolation
            self.y_old = self.y;
            self.t_old = self.t;
            self.h_old = self.h;

            // Update State
            self.k[0] = self.k[3];
            self.y = self.k[4];
            self.t = t_new;

            // Check if previous step rejected
            if let Status::RejectedStep = self.status {
                h_new = self.h.min(h_new);
                self.status = Status::Solving;
            }
        } else {
            // Step Rejected
            h_new = self.h / self.facc1.min(self.fac11 / self.safe);
            self.status = Status::RejectedStep;
        }

        // Step Complete
        self.h = constrain_step_size(h_new, self.h_min, self.h_max);
        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &V {
        &self.y
    }

    fn t_prev(&self) -> T {
        self.t_old
    }

    fn y_prev(&self) -> &V {
        &self.y_old
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        self.h = h;
    }

    fn status(&self) -> &Status<T, V, D> {
        &self.status
    }

    fn set_status(&mut self, status: Status<T, V, D>) {
        self.status = status;
    }
}

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for DOP853<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if interpolation is out of bounds
        if t_interp < self.t_old || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_old,
                t_curr: self.t,
            });
        }

        // Evaluate the interpolation polynomial at the requested time
        let s = (t_interp - self.t_old) / self.h_old;
        let s1 = T::one() - s;

        // Compute the interpolated value using nested polynomial evaluation
        let conpar = self.cont[4] + (self.cont[5] + (self.cont[6] + self.cont[7] * s) * s1) * s;

        let y_interp = self.cont[0]
            + (self.cont[1] + (self.cont[2] + (self.cont[3] + conpar * s1) * s) * s1) * s;

        Ok(y_interp)
    }
}

impl<T: Real, V: State<T>, D: CallBackData> DOP853<T, V, D> {
    /// Creates a new DOP853 NumericalMethod.
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

impl<T: Real, V: State<T>, D: CallBackData> Default for DOP853<T, V, D> {
    fn default() -> Self {
        // Convert coefficient arrays from f64 to type T
        let a = DOP853_A.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = DOP853_B.map(|x| T::from_f64(x).unwrap());
        let c = DOP853_C.map(|x| T::from_f64(x).unwrap());
        let er = DOP853_ER.map(|x| T::from_f64(x).unwrap());
        let bhh = DOP853_BHH.map(|x| T::from_f64(x).unwrap());

        let a_dense = DOP853_A_DENSE.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let c_dense = DOP853_C_DENSE.map(|x| T::from_f64(x).unwrap());
        let b_dense = DOP853_DENSE.map(|row| row.map(|x| T::from_f64(x).unwrap()));

        // Create arrays of zeros for k and cont matrices
        let k_zeros = [V::zeros(); 12];
        let cont_zeros = [V::zeros(); 8];

        DOP853 {
            // State Variables
            t: T::zero(),
            y: V::zeros(),
            h: T::zero(),

            // Settings
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
            b_dense,

            // Status and Counters
            status: Status::Uninitialized,
            h_lamb: T::zero(),
            non_stiff_counter: 0,
            stiffness_counter: 0,
            steps: 0,
            n_accepted: 0,

            // Coefficents and temporary storage
            k: k_zeros,
            y_old: V::zeros(),
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
    [
        5.260_015_195_876_773E-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.972_505_698_453_79E-2,
        5.917_517_095_361_37E-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.958_758_547_680_685E-2,
        0.0,
        8.876_275_643_042_054E-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.413_651_341_592_667E-1,
        0.0,
        -8.845_494_793_282_861E-1,
        9.248_340_032_617_92E-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.703_703_703_703_703_5E-2,
        0.0,
        0.0,
        1.708_286_087_294_738_6E-1,
        1.254_676_875_668_224_2E-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.7109375E-2,
        0.0,
        0.0,
        1.702_522_110_195_440_5E-1,
        6.021_653_898_045_596E-2,
        -1.7578125E-2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.709_200_011_850_479E-2,
        0.0,
        0.0,
        1.703_839_257_122_399_8E-1,
        1.072_620_304_463_732_8E-1,
        -1.531_943_774_862_440_2E-2,
        8.273_789_163_814_023E-3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        6.241_109_587_160_757E-1,
        0.0,
        0.0,
        -3.360_892_629_446_941_4,
        -8.682_193_468_417_26E-1,
        2.759_209_969_944_671E1,
        2.015_406_755_047_789_4E1,
        -4.348_988_418_106_996E1,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        4.776_625_364_382_643_4E-1,
        0.0,
        0.0,
        -2.488_114_619_971_667_7,
        -5.902_908_268_368_43E-1,
        2.123_005_144_818_119_3E1,
        1.527_923_363_288_242_3E1,
        -3.328_821_096_898_486E1,
        -2.033_120_170_850_862_7E-2,
        0.0,
        0.0,
        0.0,
    ],
    [
        -9.371_424_300_859_873E-1,
        0.0,
        0.0,
        5.186_372_428_844_064,
        1.091_437_348_996_729_5,
        -8.149_787_010_746_927,
        -1.852_006_565_999_696E1,
        2.273_948_709_935_050_5E1,
        2.493_605_552_679_652_3,
        -3.046_764_471_898_219_6,
        0.0,
        0.0,
    ],
    [
        2.273_310_147_516_538,
        0.0,
        0.0,
        -1.053_449_546_673_725E1,
        -2.000_872_058_224_862_5,
        -1.795_893_186_311_88E1,
        2.794_888_452_941_996E1,
        -2.858_998_277_135_023_5,
        -8.872_856_933_530_63,
        1.236_056_717_579_430_3E1,
        6.433_927_460_157_636E-1,
        0.0,
    ],
];

// C coefficients (nodes)
const DOP853_C: [f64; 12] = [
    0.0,                      // C1 (not given in constants, but must be 0)
    5.260_015_195_876_773E-2, // C2
    7.890_022_793_815_16E-2,  // C3
    1.183_503_419_072_274E-1, // C4
    2.816_496_580_927_726E-1, // C5
    3.333_333_333_333_333E-1, // C6
    0.25E+00,                 // C7
    3.076_923_076_923_077E-1, // C8
    6.512_820_512_820_513E-1, // C9
    0.6E+00,                  // C10
    8.571_428_571_428_571E-1, // C11
    1.0,                      // C12 (final point, not explicitly given)
];

// B coefficients (weights for main method)
const DOP853_B: [f64; 12] = [
    5.429_373_411_656_876_5E-2, // B1
    0.0,                        // B2-B5 are zero (not explicitly listed)
    0.0,
    0.0,
    0.0,
    4.450_312_892_752_409,      // B6
    1.891_517_899_314_500_3,    // B7
    -5.801_203_960_010_585,     // B8
    3.111_643_669_578_199E-1,   // B9
    -1.521_609_496_625_161E-1,  // B10
    2.013_654_008_040_303_4E-1, // B11
    4.471_061_572_777_259E-2,   // B12
];

// Error estimation coefficients

// Error estimation coefficients (constructed from ER values)
const DOP853_ER: [f64; 12] = [
    1.312_004_499_419_488E-2, // ER1
    0.0,                      // ER2-ER5 are zero
    0.0,
    0.0,
    0.0,
    -1.225_156_446_376_204_4,    // ER6
    -4.957_589_496_572_502E-1,   // ER7
    1.664_377_182_454_986_4,     // ER8
    -3.503_288_487_499_736_6E-1, // ER9
    3.341_791_187_130_175E-1,    // ER10
    8.192_320_648_511_571E-2,    // ER11
    -2.235_530_786_388_629_4E-2, // ER12
];

const DOP853_BHH: [f64; 3] = [
    2.440_944_881_889_764E-1,   // BHH1
    7.338_466_882_816_118E-1,   // BHH2
    2.205_882_352_941_176_6E-2, // BHH3
];

// Dense output Coefficients

// Dense output A coefficients (for the 3 extra stages used in interpolation)
const DOP853_A_DENSE: [[f64; 16]; 3] = [
    // Stage 14 coefficients (C14 = 0.1)
    [
        5.616_750_228_304_795_4E-2, // A141
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,                         // A142-A146 (zero)
        2.535_002_102_166_248_3E-1,  // A147
        -2.462_390_374_708_025E-1,   // A148
        -1.241_914_232_638_163_7E-1, // A149
        1.532_917_982_787_656_8E-1,  // A1410
        8.201_052_295_634_69E-3,     // A1411
        7.567_897_660_545_699E-3,    // A1412
        -8.298E-3,                   // A1413
        0.0,
        0.0,
        0.0, // A1414-A1416 (zero/not used)
    ],
    // Stage 15 coefficients (C15 = 0.2)
    [
        3.183_464_816_350_214E-2, // A151
        0.0,
        0.0,
        0.0,
        0.0,                        // A152-A155 (zero)
        2.830_090_967_236_677_6E-2, // A156
        5.354_198_830_743_856_6E-2, // A157
        -5.492_374_857_139_099E-2,  // A158
        0.0,
        0.0,                         // A159-A1510 (zero)
        -1.083_473_286_972_493_2E-4, // A1511
        3.825_710_908_356_584E-4,    // A1512
        -3.404_650_086_874_045_6E-4, // A1513
        1.413_124_436_746_325E-1,    // A1514
        0.0,
        0.0, // A1515-A1516 (zero/not used)
    ],
    // Stage 16 coefficients (C16 = 0.777...)
    [
        -4.288_963_015_837_919_4E-1, // A161
        0.0,
        0.0,
        0.0,
        0.0,                      // A162-A165 (zero)
        -4.697_621_415_361_164,   // A166
        7.683_421_196_062_599,    // A167
        4.068_989_818_397_11,     // A168
        3.567_271_874_552_811E-1, // A169
        0.0,
        0.0,
        0.0,                         // A1610-A1612 (zero)
        -1.399_024_165_159_014_5E-3, // A1613
        2.947_514_789_152_772_4,     // A1614
        -9.150_958_472_179_87,       // A1615
        0.0,                         // A1616 (not used)
    ],
];

const DOP853_C_DENSE: [f64; 3] = [
    0.1E+00,                  // C14
    0.2E+00,                  // C15
    7.777_777_777_777_778E-1, // C16
];

// Dense output coefficients for stage 4
const DOP853_D4: [f64; 16] = [
    -8.428_938_276_109_013, // D41
    0.0,
    0.0,
    0.0,
    0.0,                       // D42-D45 are zero
    5.667_149_535_193_777E-1,  // D46
    -3.068_949_945_949_891_7,  // D47
    2.384_667_656_512_07,      // D48
    2.117_034_582_445_028,     // D49
    -8.713_915_837_779_73E-1,  // D410
    2.240_437_430_260_788_3,   // D411
    6.315_787_787_694_688E-1,  // D412
    -8.899_033_645_133_331E-2, // D413
    1.814_850_552_085_472_7E1, // D414
    -9.194_632_392_478_356,    // D415
    -4.436_036_387_594_894,    // D416
];

// Dense output coefficients for stages 5, 6, and 7 follow same pattern
const DOP853_D5: [f64; 16] = [
    1.042_750_864_257_913_4E1, // D51
    0.0,
    0.0,
    0.0,
    0.0,                        // D52-D55 are zero
    2.422_834_917_752_581_7E2,  // D56
    1.652_004_517_172_702_8E2,  // D57
    -3.745_467_547_226_902E2,   // D58
    -2.211_366_685_312_530_6E1, // D59
    7.733_432_668_472_264,      // D510
    -3.067_408_473_108_939_8E1, // D511
    -9.332_130_526_430_229,     // D512
    1.569_723_812_177_084_5E1,  // D513
    -3.113_940_321_956_517_8E1, // D514
    -9.352_924_358_844_48,      // D515
    3.581_684_148_639_408E1,    // D516
];

const DOP853_D6: [f64; 16] = [
    1.998_505_324_200_243_3E1, // D61
    0.0,
    0.0,
    0.0,
    0.0,                        // D62-D65 are zero
    -3.870_373_087_493_518E2,   // D66
    -1.891_781_381_951_675_8E2, // D67
    5.278_081_592_054_236E2,    // D68
    -1.157_390_253_995_963E1,   // D69
    6.881_232_694_696_3,        // D610
    -1.000_605_096_691_083_8,   // D611
    7.777_137_798_053_443E-1,   // D612
    -2.778_205_752_353_508,     // D613
    -6.019_669_523_126_412E1,   // D614
    8.432_040_550_667_716E1,    // D615
    1.199_229_113_618_279E1,    // D616
];

const DOP853_D7: [f64; 16] = [
    -2.569_393_346_270_375E1, // D71
    0.0,
    0.0,
    0.0,
    0.0,                        // D72-D75 are zero
    -1.541_897_486_902_364_3E2, // D76
    -2.315_293_791_760_455E2,   // D77
    3.576_391_179_106_141E2,    // D78
    9.340_532_418_362_432E1,    // D79
    -3.745_832_313_645_163E1,   // D710
    1.040_996_495_089_623E2,    // D711
    2.984_029_342_666_05E1,     // D712
    -4.353_345_659_001_114E1,   // D713
    9.632_455_395_918_828E1,    // D714
    -3.917_726_167_561_544E1,   // D715
    -1.497_268_362_579_856_4E2, // D716
];

// Dense output coefficients as a 3D array [stage][coefficient_index]
const DOP853_DENSE: [[f64; 16]; 4] = [DOP853_D4, DOP853_D5, DOP853_D6, DOP853_D7];
