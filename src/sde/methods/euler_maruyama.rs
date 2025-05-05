//! Euler-Maruyama Method for solving stochastic differential equations.

use crate::{
    Error, Status,
    interpolate::{Interpolation, InterpolationError},
    sde::NumericalMethod,
    sde::SDE,
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
    linalg::component_multiply,
};

/// Euler-Maruyama Method for solving stochastic differential equations.
///
/// The Euler-Maruyama method is the simplest numerical method for SDEs,
/// essentially extending Euler's method to stochastic equations.
///
/// For an SDE of the form:
/// dY = a(t,Y)dt + b(t,Y)dW
///
/// The Euler-Maruyama update is:
/// Y_{n+1} = Y_n + a(t_n, Y_n)Δt + b(t_n, Y_n)ΔW_n
///
/// where ΔW_n is a Wiener process increment, produced by the SDE's noise method.
///
/// # Example
/// ```
/// use differential_equations::prelude::*;
/// use nalgebra::SVector;
/// use rand::SeedableRng;
/// use rand_distr::{Distribution, Normal};
///
/// struct GBM {
///     rng: rand::rngs::StdRng,
/// }
/// 
/// impl GBM {
///     fn new(seed: u64) -> Self {
///         Self {
///             rng: rand::rngs::StdRng::seed_from_u64(seed),
///         }
///     }
/// }
///
/// impl SDE<f64, SVector<f64, 1>> for GBM {
///     fn drift(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
///         dydt[0] = 0.1 * y[0]; // μS
///     }
///     
///     fn diffusion(&self, _t: f64, y: &SVector<f64, 1>, dydw: &mut SVector<f64, 1>) {
///         dydw[0] = 0.2 * y[0]; // σS
///     }
///     
///     fn noise(&self, dt: f64, dw: &mut SVector<f64, 1>) {
///         let normal = Normal::new(0.0, dt.sqrt()).unwrap();
///         dw[0] = normal.sample(&mut self.rng.clone());
///     }
/// }
///
/// let t0 = 0.0;
/// let tf = 1.0;
/// let y0 = SVector::<f64, 1>::new(100.0);
/// let mut solver = EM::new(0.01);
/// let gbm = GBM::new(42);
/// let gbm_problem = SDEProblem::new(gbm, t0, tf, y0);
///
/// // Solve the SDE
/// let result = gbm_problem.solve(&mut solver);
/// ```
///
pub struct EM<T: Real, V: State<T>, D: CallBackData> {
    // Step Size
    pub h: T,

    // Current State
    t: T,
    y: V,

    // Previous State
    t_prev: T,
    y_prev: V,

    // Temporary storage for derivatives
    drift: V,
    diffusion: V,

    // Status
    status: Status<T, V, D>,
}

impl<T: Real, V: State<T>, D: CallBackData> Default for EM<T, V, D> {
    fn default() -> Self {
        EM {
            h: T::from_f64(0.01).unwrap(),
            t: T::zero(),
            y: V::zeros(),
            t_prev: T::zero(),
            y_prev: V::zeros(),
            drift: V::zeros(),
            diffusion: V::zeros(),
            status: Status::Uninitialized,
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for EM<T, V, D> {
    fn init<F>(&mut self, sde: &F, t0: T, tf: T, y0: &V) -> Result<usize, Error<T, V>>
    where
        F: SDE<T, V, D>
    {
        // Check Bounds
        match validate_step_size_parameters::<T, V, D>(self.h, T::zero(), T::infinity(), t0, tf) {
            Ok(_) => {},
            Err(e) => return Err(e),
        }

        // Initialize State
        self.t = t0;
        self.y = *y0;

        // Initialize previous state
        self.t_prev = t0;
        self.y_prev = *y0;

        // Initialize derivatives
        sde.drift(t0, y0, &mut self.drift);
        sde.diffusion(t0, y0, &mut self.diffusion);

        // Initialize Status
        self.status = Status::Initialized;

        Ok(2) // 2 function evaluations: drift and diffusion
    }

    fn step<F>(&mut self, sde: &F) -> Result<usize, Error<T, V>>
    where
        F: SDE<T, V, D>
    {
        // Log previous state
        self.t_prev = self.t;
        self.y_prev = self.y;

        // Compute derivatives at current time and state
        sde.drift(self.t, &self.y, &mut self.drift);
        sde.diffusion(self.t, &self.y, &mut self.diffusion);

        // Generate noise increments using the SDE's noise method
        let mut dw = V::zeros();
        sde.noise(self.h, &mut dw);
        
        // Compute next state using Euler-Maruyama method
        let drift_term = self.drift * self.h;
        let diffusion_term = component_multiply(&self.diffusion, &dw);
        let y_new = self.y + drift_term + diffusion_term;

        // Update state
        self.t += self.h;
        self.y = y_new;

        Ok(2) // 2 function evaluations: drift and diffusion
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

    fn status(&self) -> &Status<T, V, D> {
        &self.status
    }

    fn set_status(&mut self, status: Status<T, V, D>) {
        self.status = status;
    }
}

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for EM<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, InterpolationError<T>> {
        // Check if t is within the bounds of the current step
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(InterpolationError::OutOfBounds { 
                t_interp, 
                t_prev: self.t_prev, 
                t_curr: self.t 
            });
        }

        // For stochastic methods, linear interpolation is often used as it's not easy to
        // determine the precise path between points without knowledge of the entire Wiener path
        let s = (t_interp - self.t_prev) / (self.t - self.t_prev);
        let y_interp = self.y_prev + (self.y - self.y_prev) * s;
        
        Ok(y_interp)
    }
}

impl<T: Real, V: State<T>, D: CallBackData> EM<T, V, D> {
    /// Create a new Euler-Maruyama solver with the specified step size
    ///
    /// # Arguments
    /// * `h` - Step size
    ///
    /// # Returns
    /// * A new solver instance
    pub fn new(h: T) -> Self {
        EM {
            h,
            ..Default::default()
        }
    }
}
