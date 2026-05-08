//! Dormand-Prince Runge-Kutta methods for ODEs
use crate::{
    error::Error,
    interpolate::Interpolation,
    methods::{DormandPrince, ExplicitRungeKutta, Ordinary, h_init::InitialStepSize},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize>
    OrdinaryNumericalMethod<T, Y> for ExplicitRungeKutta<Ordinary, DormandPrince, T, Y, O, S, I>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Use adaptive step size calculation for Dormand-Prince methods
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, self.order, &self.rtol, &self.atol, self.h_min, self.h_max,
                &mut evals,
            );
        }

        // Check bounds
        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = (self.filter)(h0),
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.stiffness_counter = 0;

        // Initialize State
        self.t = t0;
        self.y = y0.clone();
        self.dydt = y0.zeros_like();
        self.y_prev = y0.clone();
        self.dydt_prev = y0.zeros_like();
        self.k = core::array::from_fn(|_| y0.zeros_like());
        self.cont = core::array::from_fn(|_| y0.zeros_like());
        ode.diff(self.t, &self.y, &mut self.k[0]);
        self.dydt = self.k[0].clone();
        evals.function += 1;

        // Initialize previous state
        self.t_prev = self.t;
        self.y_prev = self.y.clone();
        self.dydt_prev = self.dydt.clone();

        // Initialize Status
        self.status = Status::Initialized;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Check if step-size is becoming too small
        if self.h.abs() < self.h_prev.abs() * T::from_f64(1e-14).unwrap() {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y.clone(),
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y.clone(),
            });
        }

        // Check max steps
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

        // Compute stages
        let mut y_stage = self.y.zeros_like();
        for i in 1..self.stages {
            y_stage = self.y.clone();

            for j in 0..i {
                y_stage.add_scaled(self.a[i][j] * self.h, &self.k[j]);
            }

            ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
        }

        // The last stage will be used for stiffness detection
        let ysti = y_stage.clone();

        // Calculate the line segment for the new y value
        let mut yseg = self.y.zeros_like();
        for i in 0..self.stages {
            yseg.add_scaled(self.b[i], &self.k[i]);
        }

        // Calculate the new y value using the line segment
        let y_new = self.y.plus_scaled(self.h, &yseg);

        // Evaluate derivative at new point for error estimation
        let t_new = self.t + self.h;

        // Number of function evaluations
        evals.function += self.stages - 1; // We already have k[0]

        // Error estimation
        let er = self.er.unwrap();
        let n = self.y.len();
        let mut err2 = T::zero();
        let mut err_state = self.y.zeros_like();
        for (j, coefficient) in er.iter().enumerate().take(self.stages) {
            err_state.add_scaled(*coefficient, &self.k[j]);
        }
        let mut err = self
            .y
            .error_norm(&y_new, &err_state, &self.atol, &self.rtol);

        if let Some(bh) = &self.bh {
            let mut err2_state = yseg.clone();
            for (j, coefficient) in bh.iter().enumerate().take(self.stages) {
                err2_state.add_scaled(-*coefficient, &self.k[j]);
            }
            err2 = self
                .y
                .error_norm(&y_new, &err2_state, &self.atol, &self.rtol);
        }
        let mut deno = err + T::from_f64(0.01).unwrap() * err2;
        if deno <= T::zero() {
            deno = T::one();
        }
        err = self.h.abs() * err * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();

        // Step size scale factor
        let order = T::from_usize(self.order).unwrap();
        let error_exponent = T::one() / order;
        let mut scale = self.safety_factor * err.powf(-error_exponent);

        // Clamp scale factor to prevent extreme step size changes
        scale = scale.max(self.min_scale).min(self.max_scale);

        // Determine if step is accepted
        if err <= T::one() {
            // Calculate the new derivative at the new point
            ode.diff(t_new, &y_new, &mut self.dydt);
            evals.function += 1;

            // stiffness detection
            let n_stiff_threshold = 100;
            if self.steps.is_multiple_of(n_stiff_threshold) {
                let stdnum = yseg.diff_norm_squared(&self.k[S - 1]);
                let stden = self.dydt.diff_norm_squared(&ysti);

                if stden > T::zero() {
                    let h_lamb = self.h * (stdnum / stden).sqrt();
                    if h_lamb > T::from_f64(6.1).unwrap() {
                        self.non_stiffness_counter = 0;
                        self.stiffness_counter += 1;
                        if self.stiffness_counter == 15 {
                            // Early Exit Stiffness Detected
                            self.status = Status::Error(Error::Stiffness {
                                t: self.t,
                                y: self.y.clone(),
                            });
                            return Err(Error::Stiffness {
                                t: self.t,
                                y: self.y.clone(),
                            });
                        }
                    }
                } else {
                    self.non_stiffness_counter += 1;
                    if self.non_stiffness_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }

            // Preparation for dense output / interpolation
            self.cont[0] = self.y.clone();
            let ydiff = y_new.minus(&self.y);
            self.cont[1] = ydiff.clone();
            let mut bspl = ydiff.zeros_like();
            bspl.add_scaled(self.h, &self.k[0]);
            bspl.add_scaled(-T::one(), &ydiff);
            self.cont[2] = bspl.clone();
            let mut cont3 = ydiff;
            cont3.add_scaled(-self.h, &self.dydt);
            cont3.add_scaled(-T::one(), &bspl);
            self.cont[3] = cont3;

            // If method has dense output stages, compute them
            if let Some(bi) = &self.bi {
                // Compute extra stages for dense output
                if I > S {
                    // First dense output coefficient, k{i=order+1}, is the derivative at the new point
                    self.k[self.stages] = self.dydt.clone();

                    for i in S + 1..I {
                        let mut y_stage = self.y.clone();
                        for j in 0..i {
                            y_stage.add_scaled(self.a[i][j] * self.h, &self.k[j]);
                        }

                        ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
                        evals.function += 1;
                    }
                }

                // Compute dense output coefficients
                for i in 4..self.order {
                    self.cont[i].fill(T::zero());
                    for j in 0..self.dense_stages {
                        self.cont[i].add_scaled(bi[i][j], &self.k[j]);
                    }
                    self.cont[i].scale_by(self.h);
                }
            }

            // For interpolation
            self.t_prev = self.t;
            self.y_prev = self.y.clone();
            self.dydt_prev = self.k[0].clone();
            self.h_prev = self.h;

            // Update the state with new values
            self.t = t_new;
            self.y = y_new;
            self.k[0] = self.dydt.clone();

            // Check if previous step is rejected
            if let Status::RejectedStep = self.status {
                self.status = Status::Solving;

                // Limit step size growth to avoid oscillations between accepted and rejected steps
                scale = scale.min(T::one());
            }
        } else {
            // Step Rejected
            self.status = Status::RejectedStep;
        }

        // Update step size
        self.h *= scale;

        // Ensure step size is within bounds
        self.h = constrain_step_size(self.h, self.h_min, self.h_max);

        // Apply step size filter
        self.h = (self.filter)(self.h);

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
        self.h = (self.filter)(h);
    }
    fn status(&self) -> &Status<T, Y> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>, const O: usize, const S: usize, const I: usize> Interpolation<T, Y>
    for ExplicitRungeKutta<Ordinary, DormandPrince, T, Y, O, S, I>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Check if interpolation is out of bounds
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // Evaluate the interpolation polynomial at the requested time
        let s = (t_interp - self.t_prev) / self.h_prev;
        let s1 = T::one() - s;

        // Functional implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
        let ilast = self.cont.len() - 1;
        let poly = (1..ilast)
            .rev()
            .fold(self.cont[ilast].clone(), |mut acc, i| {
                let factor = if i >= 4 {
                    // For the higher-order part (conpar), alternate s and s1 based on index parity
                    if (ilast - i) % 2 == 1 { s1 } else { s }
                } else {
                    // For the main polynomial part, pattern is [s1, s, s1] for indices [3, 2, 1]
                    if i % 2 == 1 { s1 } else { s }
                };
                acc.scale_by(factor);
                acc.add_scaled(T::one(), &self.cont[i]);
                acc
            });

        // Final multiplication by s for the outermost level
        let y_interp = self.cont[0].plus_scaled(s, &poly);

        Ok(y_interp)
    }
}
