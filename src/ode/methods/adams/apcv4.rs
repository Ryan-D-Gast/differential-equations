//! Adams-Predictor-Corrector 4th Order Variable Step Size Method

use super::*;
use crate::{linalg::norm, ode::methods::h_init};

///
/// Adams-Predictor-Corrector 4th Order Variable Step Size Method.
///
/// The Adams-Predictor-Corrector method is an explicit method that
/// uses the previous states to predict the next state. This implementation
/// uses a variable step size to maintain a desired accuracy.
/// It is recommended to start with a small step size so that tolerance
/// can be quickly met and the algorithm can adjust the step size accordingly.
///
/// The First 3 steps are calculated using
/// the Runge-Kutta method of order 4(5) and then the Adams-Predictor-Corrector
/// method is used to calculate the remaining steps until the final time.
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
/// use differential_equations::ode::methods::adams::APCV4;
/// use nalgebra::{SVector, vector};
///
/// struct HarmonicOscillator {
///     k: f64,
/// }
///
/// impl ODE<f64, SVector<f64, 2>> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
///         dydt[0] = y[1];
///         dydt[1] = -self.k * y[0];
///     }
/// }
/// let mut apcv4 = APCV4::new();
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// let system = HarmonicOscillator { k: 1.0 };
/// let results = ODEProblem::new(system, t0, tf, y0).solve(&mut apcv4).unwrap();
/// let expected = vector![-0.83907153, 0.54402111];
/// assert!((results.y.last().unwrap()[0] - expected[0]).abs() < 1e-6);
/// assert!((results.y.last().unwrap()[1] - expected[1]).abs() < 1e-6);
/// ```
///
/// # Settings
/// * `tol` - Tolerance
/// * `h_max` - Maximum step size
/// * `max_steps` - Maximum number of steps
///
/// ## Warning
///
/// This method is not suitable for stiff problems and can results in
/// extremely small step sizes and long computation times.
///
pub struct APCV4<T: Real, V: State<T>, D: CallBackData> {
    // Initial Step Size
    pub h0: T,

    // Current Step Size
    h: T,

    // Current State
    t: T,
    y: V,
    dydt: V,

    // Final Time
    tf: T,

    // Previous States
    t_prev: [T; 4],
    y_prev: [V; 4],

    // Previous step states
    t_old: T,
    y_old: V,
    dydt_old: V,

    // Predictor Correct Derivatives
    k1: V,
    k2: V,
    k3: V,
    k4: V,

    // Statistic Tracking
    evals: usize,
    steps: usize,

    // Status
    status: Status<T, V, D>,

    // Settings
    pub tol: T,
    pub h_max: T,
    pub h_min: T,
    pub max_steps: usize,
}

// Implement NumericalMethod Trait for APCV4
impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for APCV4<T, V, D> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        self.tf = tf;

        // Initialize initial step size if it is zero
        if self.h0 == T::zero() {
            self.h0 = h_init(
                ode, t0, tf, y0, 4, self.tol, self.tol, self.h_min, self.h_max,
            );
        }

        // Check that the initial step size is set
        match validate_step_size_parameters::<T, V, D>(self.h0, T::zero(), T::infinity(), t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize state
        self.t = t0;
        self.y = *y0;
        self.t_prev[0] = t0;
        self.y_prev[0] = *y0;

        // Previous saved steps
        self.t_old = t0;
        self.y_old = *y0;

        // Perform the first 3 steps using Runge-Kutta 4 method
        let two = T::from_f64(2.0).unwrap();
        let six = T::from_f64(6.0).unwrap();
        for i in 1..=3 {
            // Compute k1, k2, k3, k4 of Runge-Kutta 4
            ode.diff(self.t, &self.y, &mut self.k1);
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k1 * (self.h / two)),
                &mut self.k2,
            );
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k2 * (self.h / two)),
                &mut self.k3,
            );
            ode.diff(self.t + self.h, &(self.y + self.k3 * self.h), &mut self.k4);

            // Update State
            self.y += (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h / six);
            self.t += self.h;
            self.t_prev[i] = self.t;
            self.y_prev[i] = self.y;
            evals.fcn += 4; // 4 evaluations per Runge-Kutta step

            if i == 1 {
                self.dydt = self.k1;
                self.dydt_old = self.k1;
            }
        }

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
        self.steps += 1;

        // If Step size changed and it takes us to the final time perform a Runge-Kutta 4 step to finish
        if self.h != self.t_prev[0] - self.t_prev[1] && self.t + self.h == self.tf {
            let two = T::from_f64(2.0).unwrap();
            let six = T::from_f64(6.0).unwrap();

            // Perform a Runge-Kutta 4 step to finish.
            ode.diff(self.t, &self.y, &mut self.k1);
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k1 * (self.h / two)),
                &mut self.k2,
            );
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k2 * (self.h / two)),
                &mut self.k3,
            );
            ode.diff(self.t + self.h, &(self.y + self.k3 * self.h), &mut self.k4);
            evals.fcn += 4; // 4 evaluations per Runge-Kutta step

            // Update State
            self.y += (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h / six);
            self.t += self.h;
            return Ok(evals);
        }

        // Compute derivatives for history
        ode.diff(self.t_prev[3], &self.y_prev[3], &mut self.k1);
        ode.diff(self.t_prev[2], &self.y_prev[2], &mut self.k2);
        ode.diff(self.t_prev[1], &self.y_prev[1], &mut self.k3);
        ode.diff(self.t_prev[0], &self.y_prev[0], &mut self.k4);

        let predictor = self.y_prev[3]
            + (self.k1 * T::from_f64(55.0).unwrap() - self.k2 * T::from_f64(59.0).unwrap()
                + self.k3 * T::from_f64(37.0).unwrap()
                - self.k4 * T::from_f64(9.0).unwrap())
                * self.h
                / T::from_f64(24.0).unwrap();

        // Corrector step:
        ode.diff(self.t + self.h, &predictor, &mut self.k4);
        let corrector = self.y_prev[3]
            + (self.k4 * T::from_f64(9.0).unwrap() + self.k1 * T::from_f64(19.0).unwrap()
                - self.k2 * T::from_f64(5.0).unwrap()
                + self.k3 * T::from_f64(1.0).unwrap())
                * self.h
                / T::from_f64(24.0).unwrap();

        // Track number of evaluations
        evals.fcn += 5;

        // Calculate sigma for step size adjustment
        let sigma = T::from_f64(19.0).unwrap() * norm(corrector - predictor)
            / (T::from_f64(270.0).unwrap() * self.h.abs());

        // Check if Step meets tolerance
        if sigma <= self.tol {
            // Update Previous step states
            self.t_old = self.t;
            self.y_old = self.y;
            self.dydt_old = self.dydt;

            // Update state
            self.t += self.h;
            self.y = corrector;

            // Check if previous step rejected
            if let Status::RejectedStep = self.status {
                self.status = Status::Solving;
            }

            // Adjust Step Size if needed
            let two = T::from_f64(2.0).unwrap();
            let four = T::from_f64(4.0).unwrap();
            let q = (self.tol / (two * sigma)).powf(T::from_f64(0.25).unwrap());
            self.h = if q > four { four * self.h } else { q * self.h };

            // Bound Step Size
            let tf_t_abs = (self.tf - self.t).abs();
            let four_div = tf_t_abs / four;
            let h_max_effective = if self.h_max < four_div {
                self.h_max
            } else {
                four_div
            };

            self.h = constrain_step_size(self.h, self.h_min, h_max_effective);

            // Calculate Previous Steps with new step size
            self.t_prev[0] = self.t;
            self.y_prev[0] = self.y;
            let two = T::from_f64(2.0).unwrap();
            let six = T::from_f64(6.0).unwrap();
            for i in 1..=3 {
                // Compute k1, k2, k3, k4 of Runge-Kutta 4
                ode.diff(self.t, &self.y, &mut self.k1);
                ode.diff(
                    self.t + self.h / two,
                    &(self.y + self.k1 * (self.h / two)),
                    &mut self.k2,
                );
                ode.diff(
                    self.t + self.h / two,
                    &(self.y + self.k2 * (self.h / two)),
                    &mut self.k3,
                );
                ode.diff(self.t + self.h, &(self.y + self.k3 * self.h), &mut self.k4);

                // Update State
                self.y += (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h / six);
                self.t += self.h;
                self.t_prev[i] = self.t;
                self.y_prev[i] = self.y;
                self.evals += 4; // 4 evaluations per Runge-Kutta step

                if i == 1 {
                    self.dydt = self.k1;
                }
            }
        } else {
            // Step Rejected
            self.status = Status::RejectedStep;

            // Adjust Step Size
            let two = T::from_f64(2.0).unwrap();
            let tenth = T::from_f64(0.1).unwrap();
            let q = (self.tol / (two * sigma)).powf(T::from_f64(0.25).unwrap());
            self.h = if q < tenth {
                tenth * self.h
            } else {
                q * self.h
            };

            // Calculate Previous Steps with new step size
            self.t_prev[0] = self.t;
            self.y_prev[0] = self.y;
            let two = T::from_f64(2.0).unwrap();
            let six = T::from_f64(6.0).unwrap();
            for i in 1..=3 {
                // Compute k1, k2, k3, k4 of Runge-Kutta 4
                ode.diff(self.t, &self.y, &mut self.k1);
                ode.diff(
                    self.t + self.h / two,
                    &(self.y + self.k1 * (self.h / two)),
                    &mut self.k2,
                );
                ode.diff(
                    self.t + self.h / two,
                    &(self.y + self.k2 * (self.h / two)),
                    &mut self.k3,
                );
                ode.diff(self.t + self.h, &(self.y + self.k3 * self.h), &mut self.k4);

                // Update State
                self.y += (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h / six);
                self.t += self.h;
                self.t_prev[i] = self.t;
                self.y_prev[i] = self.y;
                self.evals += 4; // 4 evaluations per Runge-Kutta step
            }
        }
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
        // NumericalMethod repeats step size 4 times for each step
        // so the ODEProblem inquiring is looking for what the next
        // state will be thus the step size is multiplied by 4
        self.h * T::from_f64(4.0).unwrap()
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

// Implement the Interpolation trait for APCV4
impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for APCV4<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, InterpolationError<T>> {
        // Check if t is within the range of the solver
        if t_interp < self.t_old || t_interp > self.t {
            return Err(InterpolationError::OutOfBounds {
                t_interp,
                t_prev: self.t_old,
                t_curr: self.t,
            });
        }

        // Calculate the interpolated value using cubic Hermite interpolation
        let y_interp = cubic_hermite_interpolate(
            self.t_old,
            self.t,
            &self.y_old,
            &self.y,
            &self.dydt_old,
            &self.dydt,
            t_interp,
        );

        Ok(y_interp)
    }
}

// Initialize APCV4 with default values and chainable settings
impl<T: Real, V: State<T>, D: CallBackData> APCV4<T, V, D> {
    pub fn new() -> Self {
        APCV4 {
            ..Default::default()
        }
    }

    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    pub fn tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }

    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

impl<T: Real, V: State<T>, D: CallBackData> Default for APCV4<T, V, D> {
    fn default() -> Self {
        APCV4 {
            h0: T::zero(),
            h: T::zero(),
            t: T::zero(),
            y: V::zeros(),
            dydt: V::zeros(),
            t_prev: [T::zero(); 4],
            y_prev: [V::zeros(), V::zeros(), V::zeros(), V::zeros()],
            t_old: T::zero(),
            y_old: V::zeros(),
            dydt_old: V::zeros(),
            k1: V::zeros(),
            k2: V::zeros(),
            k3: V::zeros(),
            k4: V::zeros(),
            tf: T::zero(),
            evals: 0,
            steps: 0,
            status: Status::Uninitialized,
            tol: T::from_f64(1.0e-6).unwrap(),
            h_max: T::infinity(),
            h_min: T::zero(),
            max_steps: 1_000_000,
        }
    }
}
