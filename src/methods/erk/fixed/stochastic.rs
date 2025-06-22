//! Fixed Runge-Kutta methods for SDEs

use super::{ExplicitRungeKutta, Stochastic, Fixed};
use crate::{
    Error, Status,
    alias::Evals,
    interpolate::Interpolation,
    linalg::component_multiply,
    sde::{SDENumericalMethod, SDE},
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> SDENumericalMethod<T, V, D> for ExplicitRungeKutta<Stochastic, Fixed, T, V, D, O, S, I> {    
    fn init<F>(&mut self, sde: &mut F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size for fixed-step methods
        if self.h0 == T::zero() {
            // Simple default step size for fixed-step methods
            let duration = (tf - t0).abs();
            let default_steps = T::from_usize(100).unwrap();
            self.h0 = duration / default_steps;
        }

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.steps = 0;

        // Initialize State
        self.t = t0;
        self.y = *y0;
        
        // Calculate initial drift and diffusion
        sde.drift(self.t, &self.y, &mut self.dydt);
        let mut diffusion = V::zeros();
        sde.diffusion(self.t, &self.y, &mut diffusion);
        evals.fcn += 2; // 1 for drift + 1 for diffusion

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, sde: &mut F) -> Result<Evals, Error<T, V>>
    where
        F: SDE<T, V, D>,
    {
        let mut evals = Evals::new();

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

        // Store current state before update for interpolation
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Save k[0] as the current drift
        self.k[0] = self.dydt;

        // Compute Runge-Kutta stages for the drift term
        for i in 1..self.stages {
            let mut y_stage = self.y;

            for j in 0..i {
                y_stage += self.k[j] * (self.a[i][j] * self.h);
            }

            sde.drift(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
        }
        evals.fcn += self.stages - 1; // We already have k[0]

        // Compute deterministic part using RK weights
        let mut drift_increment = V::zeros();
        for i in 0..self.stages {
            drift_increment += self.k[i] * (self.b[i] * self.h);
        }

        // Compute diffusion term at current state
        let mut diffusion = V::zeros();
        sde.diffusion(self.t, &self.y, &mut diffusion);
        evals.fcn += 1;

        // Generate noise increments
        let mut dw = V::zeros();
        sde.noise(self.h, &mut dw);

        // Compute stochastic increment (Euler-Maruyama style)
        let diffusion_increment = component_multiply(&diffusion, &dw);

        // Combine deterministic and stochastic parts
        let y_next = self.y + drift_increment + diffusion_increment;

        // Update state
        self.t += self.h;
        self.y = y_next;
        
        // Calculate new drift for next step
        if self.fsal {
            // If FSAL (First Same As Last) is enabled, we can reuse the last derivative
            self.dydt = self.k[S - 1];
        } else {
            // Otherwise, compute the new derivative
            sde.drift(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;
        }
        
        self.status = Status::Solving;        
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

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Stochastic, Fixed, T, V, D, O, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t
            });
        }       
        
        // For stochastic methods, we typically use linear interpolation
        // since the exact path between points involves the Wiener process
        // which is not deterministic
        let s = (t_interp - self.t_prev) / (self.t - self.t_prev);
        let y_interp = self.y_prev + (self.y - self.y_prev) * s;

        Ok(y_interp)
    }
}
