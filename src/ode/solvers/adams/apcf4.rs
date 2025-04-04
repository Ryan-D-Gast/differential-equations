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
/// use differential_equations::ode::*;
/// use differential_equations::ode::solvers::APCF4;
/// use nalgebra::{SVector, vector};
/// 
/// struct HarmonicOscillator {
///     k: f64,
/// }
/// 
/// impl ODE<f64, 2, 1> for HarmonicOscillator {
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
/// let results = IVP::new(system, t0, tf, y0).solve(&mut apcf4).unwrap();
/// let expected = vector![-0.83907153, 0.54402111];
/// assert!((results.y.last().unwrap()[0] - expected[0]).abs() < 1e-2);
/// assert!((results.y.last().unwrap()[1] - expected[1]).abs() < 1e-2);
/// ```
/// 
/// # Settings
/// * `h` - Step Size
/// 
pub struct APCF4<T: Real, const R: usize, const C: usize, E: EventData> {
    // Step Size
    pub h: T,
    // Current State
    t: T,
    y: SMatrix<T, R, C>,
    dydt: SMatrix<T, R, C>,
    // Previous State for Cubic Hermite Interpolation
    t_old: T,
    y_old: SMatrix<T, R, C>,
    dydt_old: SMatrix<T, R, C>,
    // Previous States for Predictor-Corrector
    t_prev: [T; 4],
    y_prev: [SMatrix<T, R, C>; 4],
    // Predictor Correct Derivatives
    k1: SMatrix<T, R, C>, // Also the current derivative
    k2: SMatrix<T, R, C>,
    k3: SMatrix<T, R, C>,
    k4: SMatrix<T, R, C>,
    // Statistic Tracking
    evals: usize,
    steps: usize,
    // Status
    status: SolverStatus<T, R, C, E>,
}

// Implement Solver Trait for APCF4
impl<T: Real, const R: usize, const C: usize, E: EventData> Solver<T, R, C, E> for APCF4<T, R, C, E> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &SMatrix<T, R, C>) -> Result<(), SolverStatus<T, R, C, E>>
    where
        F: ODE<T, R, C, E>,
    {
        // Check Bounds
        match validate_step_size_parameters(self.h, T::zero(), T::infinity(), t0, tf) {
            Ok(h) => self.h = h,
            Err(e) => return Err(e),
        }

        // Initialize Statistics
        self.evals = 0;
        self.steps = 0;

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
            ode.diff(self.t, &self.y, &mut self.k1);
            ode.diff(self.t + self.h / two, &(self.y + self.k1 * (self.h / two)), &mut self.k2);
            ode.diff(self.t + self.h / two, &(self.y + self.k2 * (self.h / two)), &mut self.k3);
            ode.diff(self.t + self.h, &(self.y + self.k3 * self.h), &mut self.k4);

            // Update State
            self.y += (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h / six);
            self.t += self.h;
            self.t_prev[i] = self.t;
            self.y_prev[i] = self.y;
            self.evals += 4; // 4 evaluations per Runge-Kutta step

            if i == 1 {
                self.dydt = self.k1;
                self.dydt_old = self.dydt;
            }
        }

        self.status = SolverStatus::Initialized;
        Ok(())
    }

    fn step<F>(&mut self, ode: &F)
    where
        F: ODE<T, R, C, E>,
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

        let predictor = self.y_prev[3] + (
            self.k1 * T::from_f64(55.0).unwrap() - self.k2 * T::from_f64(59.0).unwrap() + self.k3 * T::from_f64(37.0).unwrap() - self.k4 * T::from_f64(9.0).unwrap()
        ) * self.h / T::from_f64(24.0).unwrap();

        // Corrector step:
        ode.diff(self.t + self.h, &predictor, &mut self.k4);
        let corrector = self.y_prev[3] + (
            self.k4 * T::from_f64(9.0).unwrap() +
            self.k1 * T::from_f64(19.0).unwrap() -
            self.k2 * T::from_f64(5.0).unwrap() +
            self.k3 * T::from_f64(1.0).unwrap()
        ) * (self.h / T::from_f64(24.0).unwrap());

        // Update state
        self.t += self.h;
        self.y = corrector;
        ode.diff(self.t, &self.y, &mut self.dydt);
        self.steps += 1;
        self.evals += 6;

        // Shift history: drop the oldest and add the new state at the end.
        self.t_prev.copy_within(1..4, 0);
        self.y_prev.copy_within(1..4, 0);
        self.t_prev[3] = self.t;
        self.y_prev[3] = self.y;
    }

    fn interpolate<F>(&mut self, _ode: &F, t: T) -> SMatrix<T, R, C>
        where 
            F: ODE<T, R, C, E> {
        cubic_hermite_interpolate(self.t_old, self.t, &self.y_old, &self.y, &self.dydt_old, &self.dydt, t)
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &SMatrix<T, R, C> {
        &self.y
    }
    
    fn dydt(&self) -> &SMatrix<T, R, C> {
        &self.dydt
    }

    fn t_prev(&self) -> T {
        self.t_old
    }

    fn y_prev(&self) -> &SMatrix<T, R, C> {
        &self.y_old
    }

    fn dydt_prev(&self) -> &SMatrix<T, R, C> {
        &self.dydt_old
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        self.h = h;
    }

    fn evals(&self) -> usize {
        self.evals
    }

    fn steps(&self) -> usize {
        self.steps
    }

    fn rejected_steps(&self) -> usize {
        0
    }

    fn accepted_steps(&self) -> usize {
        self.steps
    }

    fn status(&self) -> &SolverStatus<T, R, C, E> {
        &self.status
    }

    fn set_status(&mut self, status: SolverStatus<T, R, C, E>) {
        self.status = status;
    }
}

impl<T: Real, const R: usize, const C: usize, E: EventData> APCF4<T, R, C, E> {
    pub fn new(h: T) -> Self {
        APCF4 {
            h,
            ..Default::default()
        }
    }
}

impl<T: Real, const R: usize, const C: usize, E: EventData> Default for APCF4<T, R, C, E> {
    fn default() -> Self {
        APCF4 {
            h: T::zero(),
            t: T::zero(),
            y: SMatrix::<T, R, C>::zeros(),
            dydt: SMatrix::<T, R, C>::zeros(),
            t_prev: [T::zero(); 4],
            y_prev: [SMatrix::<T, R, C>::zeros(), SMatrix::<T, R, C>::zeros(), SMatrix::<T, R, C>::zeros(), SMatrix::<T, R, C>::zeros()],
            t_old: T::zero(),
            y_old: SMatrix::<T, R, C>::zeros(),
            dydt_old: SMatrix::<T, R, C>::zeros(),
            k1: SMatrix::<T, R, C>::zeros(),
            k2: SMatrix::<T, R, C>::zeros(),
            k3: SMatrix::<T, R, C>::zeros(),
            k4: SMatrix::<T, R, C>::zeros(),
            evals: 0,
            steps: 0,
            status: SolverStatus::Uninitialized,
        }
    }
}

