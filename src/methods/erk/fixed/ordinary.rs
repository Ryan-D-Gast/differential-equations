//! Fixed Runge-Kutta methods for ODEs

use super::{ExplicitRungeKutta, Ordinary, Fixed};
use crate::{
    Error, Status,
    alias::Evals,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    ode::{ODENumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::validate_step_size_parameters,
};

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> ODENumericalMethod<T, V, D> for ExplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {    
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
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
        }        // Initialize Statistics

        // Initialize State
        self.t = t0;
        self.y = *y0;
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.fcn += 1;

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
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

        // Save k[0] as the current derivative
        self.k[0] = self.dydt;

        // Compute stages
        for i in 1..self.stages {
            let mut y_stage = self.y;

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

        // Compute solution
        let mut y_next = self.y;
        for i in 0..self.stages {
            y_next += self.k[i] * (self.b[i] * self.h);
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

        // Update state
        self.t += self.h;
        self.y = y_next;
        
        // Calculate new derivative for next step
        if self.fsal {
            // If FSAL (First Same As Last) is enabled, we can reuse the last derivative
            self.dydt = self.k[S - 1];
        } else {
            // Otherwise, compute the new derivative
            ode.diff(self.t, &self.y, &mut self.dydt);
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

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Ordinary, Fixed, T, V, D, O, S, I> {
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

            let mut cont = [T::zero(); I];
            // Compute the interpolation coefficients using Horner's method
            for i in 0..self.dense_stages {
                // Start with the highest-order term
                cont[i] = bi[i][self.order - 1];

                // Apply Horner's method
                for j in (0..self.order - 1).rev() {
                    cont[i] = cont[i] * s + bi[i][j];
                }

                // Multiply by s
                cont[i] *= s;
            }

            // Compute the interpolated value
            let mut y_interp = self.y_prev;
            for i in 0..I {
                y_interp += self.k[i] * cont[i] * self.h_prev;
            }

            Ok(y_interp)
        } else {
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
}