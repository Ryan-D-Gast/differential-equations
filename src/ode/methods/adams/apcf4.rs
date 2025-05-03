//! Adams-Predictor-Corrector 4th Order Fixed Step Size Method.

use super::*;

///
/// Adams-Predictor-Corrector 4th Order Fixed Step Size Method.
///
/// The Adams-Predictor-Corrector method is an explicit method that
/// uses the previous states to predict the next state.
///
/// The First 3 steps, of fixed step size `h`, are calculated using
/// the Runge-Kutta method of order 4(5) and then the Adams-Predictor-Corrector
/// method is used to calculate the remaining steps tell the final time.
///
/// # Example
///
/// ```
/// use differential_equations::prelude::*;
/// use differential_equations::ode::methods::adams::APCF4;
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
/// let mut apcf4 = APCF4::new(0.01);
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0];
/// let system = HarmonicOscillator { k: 1.0 };
/// let results = ODEProblem::new(system, t0, tf, y0).solve(&mut apcf4).unwrap();
/// let expected = vector![-0.83907153, 0.54402111];
/// assert!((results.y.last().unwrap()[0] - expected[0]).abs() < 1e-2);
/// assert!((results.y.last().unwrap()[1] - expected[1]).abs() < 1e-2);
/// ```
///
/// # Settings
/// * `h` - Step Size
///
pub struct APCF4<T: Real, V: State<T>, D: CallBackData> {
    // Step Size
    pub h: T,
    // Current State
    t: T,
    y: V,
    dydt: V,
    // Previous State for Cubic Hermite Interpolation
    t_old: T,
    y_old: V,
    dydt_old: V,
    // Previous States for Predictor-Corrector
    t_prev: [T; 4],
    y_prev: [V; 4],
    // Predictor Correct Derivatives
    k1: V, // Also the current derivative
    k2: V,
    k3: V,
    k4: V,
    // Number of evaluations
    pub evals: usize,
    // Status
    status: Status<T, V, D>,
}

// Implement NumericalMethod Trait for APCF4
impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for APCF4<T, V, D> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<NumEvals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        // Check Bounds
        match validate_step_size_parameters::<T, V, D>(self.h, T::zero(), T::infinity(), t0, tf) {
            Ok(h) => self.h = h,
            Err(e) => return Err(e),
        }

        // Initialize state
        self.t = t0;
        self.y = *y0;
        self.t_prev[0] = t0;
        self.y_prev[0] = *y0;

        // Old state for interpolation
        self.t_old = self.t;
        self.y_old = self.y;

        let two = T::from_f64(2.0).unwrap();
        let six = T::from_f64(6.0).unwrap();
        let mut evals = 0;
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
            evals += 4; // 4 evaluations per Runge-Kutta step

            if i == 1 {
                self.dydt = self.k1;
                self.dydt_old = self.dydt;
            }
        }

        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<NumEvals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        // state for interpolation
        self.t_old = self.t;
        self.y_old = self.y;
        self.dydt_old = self.dydt;

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
                * (self.h / T::from_f64(24.0).unwrap());

        // Update state
        self.t += self.h;
        self.y = corrector;
        ode.diff(self.t, &self.y, &mut self.dydt);
        let evals = 6; // 6 evaluations for predictor-corrector step

        // Shift history: drop the oldest and add the new state at the end.
        self.t_prev.copy_within(1..4, 0);
        self.y_prev.copy_within(1..4, 0);
        self.t_prev[3] = self.t;
        self.y_prev[3] = self.y;
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

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for APCF4<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, InterpolationError<T>> {
        // Check if t is within bounds
        if t_interp < self.t_prev[0] || t_interp > self.t {
            return Err(InterpolationError::OutOfBounds {
                t_interp,
                t_prev: self.t_prev[0],
                t_curr: self.t,
            });
        }

        // Calculate the interpolation using cubic hermite interpolation
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

impl<T: Real, V: State<T>, D: CallBackData> APCF4<T, V, D> {
    pub fn new(h: T) -> Self {
        APCF4 {
            h,
            ..Default::default()
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> Default for APCF4<T, V, D> {
    fn default() -> Self {
        APCF4 {
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
            evals: 0,
            status: Status::Uninitialized,
        }
    }
}
