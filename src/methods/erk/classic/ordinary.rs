//! Classic Runge-Kutta methods for ODEs

use super::{ExplicitRungeKutta, Ordinary, Classic};
use crate::{
    Error, Status,
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{ODENumericalMethod, ODE, methods::h_init},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> ODENumericalMethod<T, V, D> for ExplicitRungeKutta<Ordinary, Classic, T, V, D, S, I> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Only use adaptive step size calculation if the method supports it
            if self.bh.is_some() {
                self.h0 = h_init(ode, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max);
                evals.fcn += 2;
            } else {
                // Simple default step size for fixed-step methods
                let duration = (tf - t0).abs();
                let default_steps = T::from_usize(100).unwrap();
                self.h0 = duration / default_steps;
            }
        }

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.reject = false;
        self.n_stiff = 0;

        // Initialize State
        self.t = t0;
        self.y = y0.clone();
        ode.diff(t0, y0, &mut self.dydt);
        evals.fcn += 1;

        // Initialize previous state
        self.t_prev = t0;
        self.y_prev = y0.clone();
        self.dydt_prev = self.dydt.clone();

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // Check step size
        if self.h.abs() < T::default_epsilon() {
            self.status = Status::Error(Error::StepSize {
                t: self.t, y: self.y
            });
            return Err(Error::StepSize {
                t: self.t, y: self.y
            });
        }

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

        // Save k[0] as the current derivative
        self.k[0] = self.dydt.clone();

        // Compute stages
        for i in 1..self.stages {
            let mut y_stage = self.y.clone();

            for j in 0..i {
                y_stage += self.k[j] * (self.a[i][j] * self.h);
            }

            ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
        }
        evals.fcn += self.stages - 1; // We already have k[0]

        // Store current state before update for interpolation
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.k[0];

        // For methods without embedded error estimation, simply update the solution
        if self.bh.is_none() {
            // Compute solution
            let mut y_next = self.y;
            for i in 0..self.stages {
                y_next += self.k[i] * (self.b[i] * self.h);
            }

            // Update state
            self.t += self.h;
            self.y = y_next;
            
            // Calculate new derivative for next step
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;
            
            self.status = Status::Solving;
            return Ok(evals);
        }
        
        // For adaptive methods with error estimation
        // Compute higher order solution
        let mut y_high = self.y;
        for i in 0..self.stages {
            y_high += self.k[i] * (self.b[i] * self.h);
        }

        // Compute lower order solution for error estimation
        let mut y_low = self.y;
        if let Some(bh) = &self.bh {
            for i in 0..self.stages {
                y_low += self.k[i] * (bh[i] * self.h);
            }
        }

        // Compute error estimate
        let err = y_high - y_low;

        // Calculate error norm
        let mut err_norm: T = T::zero();

        // Iterate through state elements
        for n in 0..self.y.len() {
            let tol = self.atol + self.rtol * self.y.get(n).abs().max(y_high.get(n).abs());
            err_norm = err_norm.max((err.get(n) / tol).abs());
        };

        // Determine if step is accepted
        if err_norm <= T::one() {
            // Log previous state
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.h_prev = self.h;

            if self.reject {
                self.n_stiff = 0;
                self.reject = false;
                self.status = Status::Solving;
            }

            // If method has dense output stages, compute them
            if self.bi.is_some() {
                // Compute extra stages for dense output
                for i in 0..(I - S) {
                    let mut y_stage = self.y;
                    for j in 0..self.stages + i {
                        y_stage += self.k[j] * (self.a[self.stages + i][j] * self.h);
                    }

                    ode.diff(self.t + self.c[self.stages + i] * self.h, &y_stage, &mut self.k[self.stages + i]);
                }
                evals.fcn += I - S;
            }

            // Update state with the higher-order solution
            self.t += self.h;
            self.y = y_high;
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.fcn += 1;
        } else {
            // Step rejected
            self.reject = true;
            self.status = Status::RejectedStep;
            self.n_stiff += 1;

            // Check for stiffness
            if self.n_stiff >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t, y: self.y
                });
                return Err(Error::Stiffness {
                    t: self.t, y: self.y
                });
            }
        }

        // Calculate new step size for adaptive methods
        let order = T::from_usize(self.order).unwrap();
        let err_order = T::one() / order;

        // Step size controller
        let scale = self.safety_factor * err_norm.powf(-err_order);
        let scale = scale.max(self.min_scale).min(self.max_scale);
        self.h *= scale;

        // Ensure step size is within bounds
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);
        
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

impl<T: Real, V: State<T>, D: CallBackData, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Ordinary, Classic, T, V, D, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t
            });
        }

        // If method has dense output coefficients, use them
        if self.bi.is_some() {
            // Calculate the normalized distance within the step [0, 1]
            let s = (t_interp - self.t_prev) / self.h_prev;
            
            // Get the interpolation coefficients
            let bi = self.bi.as_ref().unwrap();

            // Compute the interpolation coefficients using Horner's method
            for i in 0..self.dense_stages {
                // Start with the highest-order term
                self.cont[i] = bi[i][self.dense_stages - 1];

                // Apply Horner's method
                for j in (0..self.dense_stages - 1).rev() {
                    self.cont[i] = self.cont[i] * s + bi[i][j];
                }

                // Multiply by s
                self.cont[i] *= s;
            }

            // Compute the interpolated value
            let mut y_interp = self.y_prev;
            for i in 0..I {
                if i < self.k.len() && i < self.cont.len() {
                    y_interp += self.k[i] * (self.cont[i] * self.h_prev);
                }
            }

            return Ok(y_interp);
        }
        
        // Otherwise use cubic Hermite interpolation
        let y_interp = cubic_hermite_interpolate(
            self.t_prev, 
            self.t, 
            &self.y_prev, 
            &self.y, 
            &self.dydt_prev, 
            &self.dydt, 
            t_interp
        );

        Ok(y_interp)
    }
}