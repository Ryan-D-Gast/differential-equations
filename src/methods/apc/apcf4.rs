//! Adams-Predictor-Corrector 4th Order Fixed Step Size Method.

use super::AdamsPredictorCorrector;
use crate::{
    Error, Status,
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{OrdinaryNumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::{validate_step_size_parameters},
    methods::{Ordinary, Fixed},
};

impl<T: Real, V: State<T>, D: CallBackData> AdamsPredictorCorrector<Ordinary, Fixed, T, V, D, 4> {
    /// Adams-Predictor-Corrector 4th Order Fixed Step Size Method.
    ///
    /// The Adams-Predictor-Corrector method is an explicit method that
    /// uses the previous states to predict the next state.
    ///
    /// The First 3 steps, of fixed step size `h`, are calculated using
    /// the Runge-Kutta method of order 4(5) and then the Adams-Predictor-Corrector
    /// method is used to calculate the remaining steps until the final time.
    ///
    /// # Example
    ///
    /// ```
    /// use differential_equations::prelude::*;
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
    /// let mut apcf4 = AdamsPredictorCorrector::f4(0.01);
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
    pub fn f4(h: T) -> Self {
        Self {
            h,
            ..Default::default()
        }
    }
}

// Implement OrdinaryNumericalMethod Trait for APCF4
impl<T: Real, V: State<T>, D: CallBackData> OrdinaryNumericalMethod<T, V, D> for AdamsPredictorCorrector<Ordinary, Fixed, T, V, D, 4> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

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
        for i in 1..=3 {
            // Compute k1, k2, k3, k4 of Runge-Kutta 4
            ode.diff(self.t, &self.y, &mut self.k[0]);
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k[0] * (self.h / two)),
                &mut self.k[1],
            );
            ode.diff(
                self.t + self.h / two,
                &(self.y + self.k[1] * (self.h / two)),
                &mut self.k[2],
            );
            ode.diff(self.t + self.h, &(self.y + self.k[2] * self.h), &mut self.k[3]);

            // Update State
            self.y += (self.k[0] + self.k[1] * two + self.k[2] * two + self.k[3]) * (self.h / six);
            self.t += self.h;
            self.t_prev[i] = self.t;
            self.y_prev[i] = self.y;
            evals.fcn += 4; // 4 evaluations per Runge-Kutta step

            if i == 1 {
                self.dydt = self.k[0];
                self.dydt_old = self.dydt;
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

        // state for interpolation
        self.t_old = self.t;
        self.y_old = self.y;
        self.dydt_old = self.dydt;

        // Compute derivatives for history
        ode.diff(self.t_prev[3], &self.y_prev[3], &mut self.k[0]);
        ode.diff(self.t_prev[2], &self.y_prev[2], &mut self.k[1]);
        ode.diff(self.t_prev[1], &self.y_prev[1], &mut self.k[2]);
        ode.diff(self.t_prev[0], &self.y_prev[0], &mut self.k[3]);

        let predictor = self.y_prev[3]
            + (self.k[0] * T::from_f64(55.0).unwrap() - self.k[1] * T::from_f64(59.0).unwrap()
                + self.k[2] * T::from_f64(37.0).unwrap()
                - self.k[3] * T::from_f64(9.0).unwrap())
                * self.h
                / T::from_f64(24.0).unwrap();

        // Corrector step:
        ode.diff(self.t + self.h, &predictor, &mut self.k[3]);
        let corrector = self.y_prev[3]
            + (self.k[3] * T::from_f64(9.0).unwrap() + self.k[0] * T::from_f64(19.0).unwrap()
                - self.k[1] * T::from_f64(5.0).unwrap()
                + self.k[2] * T::from_f64(1.0).unwrap())
                * (self.h / T::from_f64(24.0).unwrap());

        // Update state
        self.t += self.h;
        self.y = corrector;
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.fcn += 6; // 6 evaluations for predictor-corrector step

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

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for AdamsPredictorCorrector<Ordinary, Fixed, T, V, D, 4> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if t is within bounds
        if t_interp < self.t_prev[0] || t_interp > self.t {
            return Err(Error::OutOfBounds {
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