//! Derivative-Free Milstein method for Stochastic Differential Equations

use crate::{
    error::Error,
    interpolate::{Interpolation, linear_interpolate},
    linalg::{component_multiply, component_square},
    sde::{SDE, StochasticNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::validate_step_size_parameters,
};

/// Derivative-Free Milstein method for solving SDEs.
///
/// Provides strong order 1.0 convergence for commutative/diagonal noise,
/// which is an improvement over the 0.5 strong order of Euler-Maruyama.
pub struct Milstein<T: Real, Y: State<T>> {
    pub h0: T,
    h: T,
    t: T,
    y: Y,
    t_prev: T,
    y_prev: Y,
    dydt: Y,

    // Settings
    pub h_min: T,
    pub h_max: T,
    pub max_steps: usize,

    // Statistics
    steps: usize,
    status: Status<T, Y>,
}

impl<T: Real, Y: State<T>> Milstein<T, Y> {
    /// Creates a new Milstein method solver
    pub fn new(h0: T) -> Self {
        Self {
            h0,
            h: h0,
            t: T::zero(),
            y: Y::zeros(),
            t_prev: T::zero(),
            y_prev: Y::zeros(),
            dydt: Y::zeros(),
            h_min: T::zero(),
            h_max: T::infinity(),
            max_steps: 10_000,
            steps: 0,
            status: Status::Uninitialized,
        }
    }

    /// Set minimum step size
    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set maximum step size
    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    /// Set maximum number of steps
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

impl<T: Real, Y: State<T>> StochasticNumericalMethod<T, Y> for Milstein<T, Y> {
    fn init<F>(&mut self, sde: &mut F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: SDE<T, Y>,
    {
        let mut evals = Evals::new();

        if self.h0 == T::zero() {
            let duration = (tf - t0).abs();
            self.h0 = duration / T::from_f64(100.0).unwrap();
        }

        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        self.steps = 0;
        self.t = t0;
        self.y = y0.clone();
        self.dydt = y0.zeros_like();
        self.t_prev = t0;
        self.y_prev = y0.clone();

        sde.drift(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, sde: &mut F) -> Result<Evals, Error<T, Y>>
    where
        F: SDE<T, Y>,
    {
        let mut evals = Evals::new();

        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y.clone(),
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y.clone(),
            });
        }
        self.steps += 1;

        self.t_prev = self.t;
        self.y_prev = self.y.clone();

        // Calculate diffusion at y_n
        let mut diffusion = self.y.zeros_like();
        sde.diffusion(self.t, &self.y, &mut diffusion);
        evals.function += 1;

        // Generate noise increments
        let mut dw = self.y.zeros_like();
        sde.noise(self.h, &mut dw);

        // Derivative-free Milstein correction
        // y_aux = y_n + b(t_n, y_n) * sqrt(h)
        let sqrt_h = self.h.sqrt();
        let mut y_aux = self.y.clone();
        y_aux.add_scaled(sqrt_h, &diffusion);

        // b_aux = b(t_n, y_aux)
        let mut diffusion_aux = self.y.zeros_like();
        sde.diffusion(self.t, &y_aux, &mut diffusion_aux);
        evals.function += 1;

        // term = (b_aux - b) * (dw^2 - h) / (2 * sqrt(h))
        let dw_sq = component_square(&dw);
        let mut milstein_term = self.y.zeros_like();
        let factor = T::one() / (T::from_f64(2.0).unwrap() * sqrt_h);

        for i in 0..self.y.len() {
            let diff = diffusion_aux.get_component(i) - diffusion.get_component(i);
            let dws_minus_h = dw_sq.get_component(i) - self.h;
            milstein_term.set_component(i, diff * dws_minus_h * factor);
        }

        // Combine deterministic, Euler-Maruyama, and Milstein parts
        let mut drift_increment = self.dydt.clone();
        drift_increment.scale_mut(self.h);

        let diffusion_increment = component_multiply(&diffusion, &dw);

        let y_next = self.y.plus_linear_combination(&[
            (&drift_increment, T::one()),
            (&diffusion_increment, T::one()),
            (&milstein_term, T::one()),
        ]);

        self.t += self.h;
        self.y = y_next;

        // Drift for next step
        sde.drift(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        self.status = Status::Solving;
        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }
    fn y(&self) -> &Y {
        &self.y
    }
    fn t_prev(&self) -> T {
        self.t_prev
    }
    fn y_prev(&self) -> &Y {
        &self.y_prev
    }
    fn h(&self) -> T {
        self.h
    }
    fn set_h(&mut self, h: T) {
        self.h = h;
    }
    fn status(&self) -> &Status<T, Y> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>> Interpolation<T, Y> for Milstein<T, Y> {
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }
        Ok(linear_interpolate(
            self.t_prev,
            self.t,
            &self.y_prev,
            &self.y,
            t_interp,
        ))
    }
}
