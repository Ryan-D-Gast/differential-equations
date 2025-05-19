//! Runge-Kutta-Maruyama Method for solving stochastic differential equations.

use crate::{
    Error, Status,
    interpolate::{Interpolation, InterpolationError},
    linalg::component_multiply,
    sde::{NumericalMethod, SDE},
    alias::Evals,
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

/// Runge-Kutta-Maruyama Method for solving stochastic differential equations.
///
/// This method extends the fourth-order Runge-Kutta method for deterministic ODEs
/// to the stochastic case. It uses RK4 for the drift term to achieve higher-order
/// accuracy in the deterministic part, while maintaining the Euler-Maruyama approach
/// for the diffusion term.
///
/// For an SDE of the form:
/// dY = a(t,Y)dt + b(t,Y)dW
///
/// The method uses RK4 for the deterministic part and Euler-Maruyama for the stochastic part:
/// k1 = a(t_n, Y_n)
/// k2 = a(t_n + h/2, Y_n + h·k1/2)
/// k3 = a(t_n + h/2, Y_n + h·k2/2)
/// k4 = a(t_n + h, Y_n + h·k3)
/// Y_{n+1} = Y_n + (h/6)·(k1 + 2k2 + 2k3 + k4) + b(t_n, Y_n)·ΔW_n
///
/// Note: While this method has higher precision for the deterministic part,
/// it remains order 0.5 for strong convergence of the full SDE due to the
/// stochastic integration limitations.
///
/// # Example
/// ```
/// use differential_equations::prelude::*;
/// use nalgebra::SVector;
/// use rand::SeedableRng;
/// use rand_distr::{Distribution, Normal};
///
/// struct OrnsteinUhlenbeck {
///     theta: f64,  // Mean reversion speed
///     mu: f64,     // Long-term mean
///     sigma: f64,  // Volatility
///     rng: rand::rngs::StdRng,
/// }
///
/// impl OrnsteinUhlenbeck {
///     fn new(theta: f64, mu: f64, sigma: f64, seed: u64) -> Self {
///         Self {
///             theta,
///             mu,
///             sigma,
///             rng: rand::rngs::StdRng::seed_from_u64(seed),
///         }
///     }
/// }
///
/// impl SDE<f64, SVector<f64, 1>> for OrnsteinUhlenbeck {
///     fn drift(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
///         // Mean-reverting drift: θ(μ-y)
///         dydt[0] = self.theta * (self.mu - y[0]);
///     }
///     
///     fn diffusion(&self, _t: f64, _y: &SVector<f64, 1>, dydw: &mut SVector<f64, 1>) {
///         // Constant volatility
///         dydw[0] = self.sigma;
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
/// let y0 = SVector::<f64, 1>::new(2.0);  // Initial value away from mean
/// let ou_process = OrnsteinUhlenbeck::new(0.5, 1.0, 0.1, 42);
/// let mut solver = RKM4::new(0.01);
/// let ou_problem = SDEProblem::new(ou_process, t0, tf, y0);
///
/// // Solve the SDE
/// let result = ou_problem.solve(&mut solver);
/// ```
///
pub struct RKM4<T: Real, V: State<T>, D: CallBackData> {
    // Step Size
    pub h: T,

    // Current State
    t: T,
    y: V,

    // Previous State
    t_prev: T,
    y_prev: V,

    // Temporary storage for derivatives and intermediate steps
    k1: V,
    k2: V,
    k3: V,
    k4: V,
    y_temp: V,
    diffusion: V,

    // Status
    status: Status<T, V, D>,
}

impl<T: Real, V: State<T>, D: CallBackData> Default for RKM4<T, V, D> {
    fn default() -> Self {
        RKM4 {
            h: T::from_f64(0.01).unwrap(),
            t: T::zero(),
            y: V::zeros(),
            t_prev: T::zero(),
            y_prev: V::zeros(),
            k1: V::zeros(),
            k2: V::zeros(),
            k3: V::zeros(),
            k4: V::zeros(),
            y_temp: V::zeros(),
            diffusion: V::zeros(),
            status: Status::Uninitialized,
        }
    }
}

impl<T: Real, V: State<T>, D: CallBackData> NumericalMethod<T, V, D> for RKM4<T, V, D> {
    fn init<F>(&mut self, sde: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check Bounds
        match validate_step_size_parameters::<T, V, D>(self.h, T::zero(), T::infinity(), t0, tf) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }

        // Initialize State
        self.t = t0;
        self.y = *y0;

        // Initialize previous state
        self.t_prev = t0;
        self.y_prev = *y0;

        // Initialize derivatives (calculate first k1 and diffusion coefficient)
        sde.drift(t0, y0, &mut self.k1);
        sde.diffusion(t0, y0, &mut self.diffusion);
        evals.fcn += 2; // 2 function evaluation for diffusion and drift

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, sde: &F) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Log previous state
        self.t_prev = self.t;
        self.y_prev = self.y;

        // RK4 for the deterministic part (drift)

        // k1 is already calculated from previous step or init
        sde.drift(self.t, &self.y, &mut self.k1);

        // k2 calculation: k2 = f(t + h/2, y + h*k1/2)
        let half_h = self.h / T::from_f64(2.0).unwrap();
        let t_half = self.t + half_h;

        // y_temp = y + h*k1/2
        self.y_temp = self.y + self.k1 * half_h;
        sde.drift(t_half, &self.y_temp, &mut self.k2);

        // k3 calculation: k3 = f(t + h/2, y + h*k2/2)
        // y_temp = y + h*k2/2
        self.y_temp = self.y + self.k2 * half_h;
        sde.drift(t_half, &self.y_temp, &mut self.k3);

        // k4 calculation: k4 = f(t + h, y + h*k3)
        let t_full = self.t + self.h;
        // y_temp = y + h*k3
        self.y_temp = self.y + self.k3 * self.h;
        sde.drift(t_full, &self.y_temp, &mut self.k4);

        // Compute diffusion term at the current time and state
        sde.diffusion(self.t, &self.y, &mut self.diffusion);

        // Generate noise increments using the SDE's noise method
        let mut dw = V::zeros();
        sde.noise(self.h, &mut dw);

        evals.fcn += 5; // 4 for drift (k1, k2, k3, k4) + 1 for diffusion

        // Compute next state using the Runge-Kutta-Maruyama formula
        // y_new = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4) + diffusion * dW

        // Calculate RK4 weighted sum for deterministic part
        let sixth = T::from_f64(1.0 / 6.0).unwrap();
        let two = T::from_f64(2.0).unwrap();

        let drift_term = (self.k1 + self.k2 * two + self.k3 * two + self.k4) * (self.h * sixth);

        // Calculate stochastic part (Euler-Maruyama style)
        let diffusion_term = component_multiply(&self.diffusion, &dw);

        // Combine deterministic and stochastic components
        let y_new = self.y + drift_term + diffusion_term;

        // Update state
        self.t = t_full;
        self.y = y_new;

        Ok(evals) // 5 function evaluations: 4 for drift (RK4) + 1 for diffusion
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

impl<T: Real, V: State<T>, D: CallBackData> Interpolation<T, V> for RKM4<T, V, D> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, InterpolationError<T>> {
        // Check if t is within the bounds of the current step
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(InterpolationError::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // For stochastic methods, linear interpolation is often used as it's not easy to
        // determine the precise path between points without knowledge of the entire Wiener path
        let s = (t_interp - self.t_prev) / (self.t - self.t_prev);
        let y_interp = self.y_prev + (self.y - self.y_prev) * s;

        Ok(y_interp)
    }
}

impl<T: Real, V: State<T>, D: CallBackData> RKM4<T, V, D> {
    /// Create a new Runge-Kutta-Maruyama solver with the specified step size
    ///
    /// # Arguments
    /// * `h` - Step size
    ///
    /// # Returns
    /// * A new solver instance
    pub fn new(h: T) -> Self {
        RKM4 {
            h,
            ..Default::default()
        }
    }
}
