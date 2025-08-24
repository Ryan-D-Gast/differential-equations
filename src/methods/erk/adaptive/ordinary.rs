//! Adaptive Runge-Kutta methods for ODEs

use crate::{
    error::Error,
    interpolate::{Interpolation, cubic_hermite_interpolate},
    methods::{Adaptive, ExplicitRungeKutta, Ordinary, h_init::InitialStepSize},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y, D> for ExplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Only use adaptive step size calculation if the method supports it
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, self.order, &self.rtol, &self.atol, self.h_min, self.h_max,
                &mut evals,
            );
            evals.function += 2;
        }

        // Check bounds
        match validate_step_size_parameters::<T, Y, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.stiffness_counter = 0;

        // Initialize State
        self.t = t0;
        self.y = *y0;
        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Check step size
        if self.h.abs() < self.h_prev.abs() * T::from_f64(1e-14).unwrap() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y,
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y,
            });
        }

        // Check max steps
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
        evals.function += self.stages - 1; // We already have k[0]

        // For adaptive methods with error estimation
        // Compute higher order solution
        let mut y_high = self.y;
        for i in 0..self.stages {
            y_high += self.k[i] * (self.b[i] * self.h);
        }

        // Compute lower order solution for error estimation
        let mut y_low = self.y;
        let bh = &self.bh.unwrap();
        for i in 0..self.stages {
            y_low += self.k[i] * (bh[i] * self.h);
        }

        // Compute error estimate
        let err = y_high - y_low;

        // Calculate error norm
        let mut err_norm: T = T::zero();

        // Iterate through state elements
        for n in 0..self.y.len() {
            let tol = self.atol[n] + self.rtol[n] * self.y.get(n).abs().max(y_high.get(n).abs());
            err_norm = err_norm.max((err.get(n) / tol).abs());
        }

        // Step size scale factor
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err_norm.powf(-error_exponent);

        // Clamp scale factor to prevent extreme step size changes
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Determine if step is accepted
        if err_norm <= T::one() {
            // Log previous state
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.k[0];
            self.h_prev = self.h;

            if let Status::RejectedStep = self.status {
                self.stiffness_counter = 0;
                self.status = Status::Solving;

                // Limit step size growth to avoid oscillations between accepted and rejected steps
                scale = scale.min(T::one());
            }

            // If method has dense output stages, compute them
            if self.bi.is_some() {
                // Compute extra stages for dense output
                for i in 0..(I - S) {
                    let mut y_stage = self.y;
                    for j in 0..self.stages + i {
                        y_stage += self.k[j] * (self.a[self.stages + i][j] * self.h);
                    }

                    ode.diff(
                        self.t + self.c[self.stages + i] * self.h,
                        &y_stage,
                        &mut self.k[self.stages + i],
                    );
                }
                evals.function += I - S;
            }

            // Update state with the higher-order solution
            self.t += self.h;
            self.y = y_high;

            // Compute the derivative for the next step
            if self.fsal {
                // If FSAL (First Same As Last) is enabled, we can reuse the last derivative
                self.dydt = self.k[S - 1];
            } else {
                // Otherwise, compute the new derivative
                ode.diff(self.t, &self.y, &mut self.dydt);
                evals.function += 1;
            }
        } else {
            // Step rejected
            self.status = Status::RejectedStep;
            self.stiffness_counter += 1;

            // Check for stiffness
            if self.stiffness_counter >= self.max_rejects {
                self.status = Status::Error(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
                return Err(Error::Stiffness {
                    t: self.t,
                    y: self.y,
                });
            }
        }

        // Update step size
        self.h *= scale;

        // Ensure step size is within bounds
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);

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
    fn status(&self) -> &Status<T, Y, D> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y, D>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize>
    Interpolation<T, Y> for ExplicitRungeKutta<Ordinary, Adaptive, T, Y, D, O, S, I>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Check if t is within bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
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
                t_interp,
            );

            Ok(y_interp)
        }
    }
}
