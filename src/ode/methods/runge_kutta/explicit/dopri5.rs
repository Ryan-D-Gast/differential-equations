//! DOPRI5 NumericalMethod for Ordinary Differential Equations.

use crate::{
    Error, Status,
    interpolate::{Interpolation, InterpolationError},
    ode::{NumericalMethod, ODE, methods::h_init},
    traits::{CallBackData, Real, State},
    alias::NumEvals,
    utils::{constrain_step_size, validate_step_size_parameters},
};

/// Dormand Prince 5(4) Method for solving ordinary differential equations.
/// 5th order Dormand Prince method with embedded 4th order error estimation and
/// dense output interpolation.
///
/// # Example
/// ```
/// use differential_equations::prelude::*;
/// use nalgebra::{SVector, vector};
///
/// let mut dopri5 = DOPRI5::new()
///    .rtol(1e-6)
///    .atol(1e-6);
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
/// let solution = ODEProblem::new(Example, t0, tf, y0).solve(&mut dopri5).unwrap();
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
/// * `max_steps` - 100_000
/// * `n_stiff` - 1000
/// * `safe`   - 0.9
/// * `fac1`   - 0.2
/// * `fac2`   - 10.0
/// * `beta`   - 0.04
///
pub struct DOPRI5<T: Real, V: State<T>, D: CallBackData> {
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

    // DOPRI5 Specific Settings
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
    a: [[T; 7]; 7],
    b: [T; 7],
    c: [T; 7],
    er: [T; 7],

    // Dense output coefficients
    d: [T; 7],

    // Derivatives - using array instead of individually numbered variables
    k: [V; 7], // k[0] is derivative at t, others are stage derivatives

    // For Interpolation - using array instead of individually numbered variables
    y_old: V,     // State at Previous Step
    t_old: T,     // Time of Previous Step
    h_old: T,     // Step Size of Previous Step
    cont: [V; 5], // Interpolation coefficients
}

impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for DOPRI5<T, V, D> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<NumEvals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        // Set Current State as Initial State
        self.t = t0;
        self.y = *y0;

        // Calculate derivative at t0
        ode.diff(t0, y0, &mut self.k[0]);
        let mut evals = 1; // Increment function evaluations for initial derivative calculation

        // Initialize Previous State
        self.t_old = self.t;
        self.y_old = self.y;

        // Calculate Initial Step
        if self.h0 == T::zero() {
            self.h0 = h_init(
                ode, t0, tf, y0, 5, self.rtol, self.atol, self.h_min, self.h_max,
            );
            evals += 1; // Increment function evaluations for initial step size calculation

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

    fn step<F>(&mut self, ode: &F) -> Result<NumEvals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
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

        // The six stages
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
            &(self.y
                + self.k[0] * (self.a[3][0] * self.h)
                + self.k[1] * (self.a[3][1] * self.h)
                + self.k[2] * (self.a[3][2] * self.h)),
            &mut self.k[3],
        );
        ode.diff(
            self.t + self.c[4] * self.h,
            &(self.y
                + self.k[0] * (self.a[4][0] * self.h)
                + self.k[1] * (self.a[4][1] * self.h)
                + self.k[2] * (self.a[4][2] * self.h)
                + self.k[3] * (self.a[4][3] * self.h)),
            &mut self.k[4],
        );
        ode.diff(
            self.t + self.c[5] * self.h,
            &(self.y
                + self.k[0] * (self.a[5][0] * self.h)
                + self.k[1] * (self.a[5][1] * self.h)
                + self.k[2] * (self.a[5][2] * self.h)
                + self.k[3] * (self.a[5][3] * self.h)
                + self.k[4] * (self.a[5][4] * self.h)),
            &mut self.k[5],
        );

        let ysti = self.y
            + self.k[0] * (self.a[6][0] * self.h)
            + self.k[2] * (self.a[6][2] * self.h)
            + self.k[3] * (self.a[6][3] * self.h)
            + self.k[4] * (self.a[6][4] * self.h)
            + self.k[5] * (self.a[6][5] * self.h);

        let t_new = self.t + self.h;
        ode.diff(t_new, &ysti, &mut self.k[6]);

        let y_new = self.y
            + self.k[0] * (self.b[0] * self.h)
            + self.k[2] * (self.b[2] * self.h)
            + self.k[3] * (self.b[3] * self.h)
            + self.k[4] * (self.b[4] * self.h)
            + self.k[5] * (self.b[5] * self.h)
            + self.k[6] * (self.b[6] * self.h);

        ode.diff(t_new, &y_new, &mut self.k[1]);

        // Calculate error using embedded method
        let mut err = T::zero();

        let n = self.y.len();
        for i in 0..n {
            let sk = self.atol + self.rtol * self.y.get(i).abs().max(y_new.get(i).abs());
            let erri = self.h
                * (self.er[0] * self.k[0].get(i)
                    + self.er[2] * self.k[2].get(i)
                    + self.er[3] * self.k[3].get(i)
                    + self.er[4] * self.k[4].get(i)
                    + self.er[5] * self.k[5].get(i)
                    + self.er[6] * self.k[6].get(i));
            err += (erri / sk).powi(2);
        }
        err = (err / T::from_usize(n).unwrap()).sqrt();

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
            self.n_accepted += 1;

            // stiffness detection
            if self.n_accepted % self.n_stiff == 0 || self.stiffness_counter > 0 {
                let mut stnum = T::zero();
                let mut stden = T::zero();

                for i in 0..n {
                    let stnum_i = self.k[1].get(i) - self.k[6].get(i);
                    stnum += stnum_i * stnum_i;

                    let stden_i = y_new.get(i) - ysti.get(i);
                    stden += stden_i * stden_i;
                }

                if stden > T::zero() {
                    self.h_lamb = self.h * (stnum / stden).sqrt();
                }

                if self.h_lamb > T::from_f64(3.25).unwrap() {
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

            // Prepare for dense output / interpolation
            // Store data for interpolation
            let ydiff = y_new - self.y;
            let bspl = self.k[0] * self.h - ydiff;

            self.cont[0] = self.y;
            self.cont[1] = ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.k[1] * self.h - bspl;

            // Compute the dense output coefficient
            self.cont[4] = (self.k[0] * self.d[0]
                + self.k[2] * self.d[2]
                + self.k[3] * self.d[3]
                + self.k[4] * self.d[4]
                + self.k[5] * self.d[5]
                + self.k[6] * self.d[6])
                * self.h;

            // For Interpolation
            self.y_old = self.y;
            self.t_old = self.t;
            self.h_old = self.h;

            // Update State
            self.k[0] = self.k[1];
            self.y = y_new;
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
        Ok(7)
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

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for DOPRI5<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, InterpolationError<T>> {
        // Check if interpolation is out of bounds
        if t_interp < self.t_old || t_interp > self.t {
            return Err(InterpolationError::OutOfBounds {
                t_interp,
                t_prev: self.t_old,
                t_curr: self.t,
            });
        }

        // Evaluate the interpolation polynomial at the requested time
        let s = (t_interp - self.t_old) / self.h_old;
        let s1 = T::one() - s;

        // Use the provided dense output formula
        let y_interp = self.cont[0]
            + (self.cont[1] + (self.cont[2] + (self.cont[3] + self.cont[4] * s1) * s) * s1) * s;

        Ok(y_interp)
    }
}

impl<T: Real, V: State<T>, D: CallBackData> DOPRI5<T, V, D> {
    /// Creates a new DOPRI5 NumericalMethod.
    ///
    /// # Returns
    /// * DOPRI5 Struct ready to go for solving.
    ///  
    pub fn new() -> Self {
        DOPRI5 {
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

impl<T: Real, V: State<T>, D: CallBackData> Default for DOPRI5<T, V, D> {
    fn default() -> Self {
        // Convert coefficient arrays from f64 to type T
        let a = DOPRI5_A.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = DOPRI5_B.map(|x| T::from_f64(x).unwrap());
        let c = DOPRI5_C.map(|x| T::from_f64(x).unwrap());
        let er = DOPRI5_E.map(|x| T::from_f64(x).unwrap());
        let d = DOPRI5_D.map(|x| T::from_f64(x).unwrap());

        // Create arrays of zeros for k and cont matrices
        let k_zeros = [V::zeros(); 7];
        let cont_zeros = [V::zeros(); 5];

        DOPRI5 {
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
            max_steps: 100_000,
            n_stiff: 1000,
            safe: T::from_f64(0.9).unwrap(),
            fac1: T::from_f64(0.2).unwrap(),
            fac2: T::from_f64(10.0).unwrap(),
            beta: T::from_f64(0.04).unwrap(),
            expo1: T::from_f64(1.0 / 5.0).unwrap(),
            facc1: T::from_f64(1.0 / 0.2).unwrap(),
            facc2: T::from_f64(1.0 / 10.0).unwrap(),
            facold: T::from_f64(1.0e-4).unwrap(),
            fac11: T::zero(),
            fac: T::zero(),

            // Butcher Tableau Coefficients
            a,
            b,
            c,
            er,
            d,

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

// DOPRI5 Butcher Tableau

// A matrix (7x7, lower triangular)
const DOPRI5_A: [[f64; 7]; 7] = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
];

// C coefficients (nodes)
const DOPRI5_C: [f64; 7] = [
    0.0,       // C1
    0.2,       // C2
    0.3,       // C3
    0.8,       // C4
    8.0 / 9.0, // C5
    1.0,       // C6
    1.0,       // C7
];

// B coefficients (weights for main method)
const DOPRI5_B: [f64; 7] = [
    35.0 / 384.0,     // B1
    0.0,              // B2
    500.0 / 1113.0,   // B3
    125.0 / 192.0,    // B4
    -2187.0 / 6784.0, // B5
    11.0 / 84.0,      // B6
    0.0,              // B7
];

// Error estimation coefficients
const DOPRI5_E: [f64; 7] = [
    71.0 / 57600.0,      // E1
    0.0,                 // E2
    -71.0 / 16695.0,     // E3
    71.0 / 1920.0,       // E4
    -17253.0 / 339200.0, // E5
    22.0 / 525.0,        // E6
    -1.0 / 40.0,         // E7
];

// Dense output coefficients
const DOPRI5_D: [f64; 7] = [
    -12715105075.0 / 11282082432.0,  // D1
    0.0,                             // D2
    87487479700.0 / 32700410799.0,   // D3
    -10690763975.0 / 1880347072.0,   // D4
    701980252875.0 / 199316789632.0, // D5
    -1453857185.0 / 822651844.0,     // D6
    69997945.0 / 29380423.0,         // D7
];
