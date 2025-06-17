//! Dormand-Prince Runge-Kutta methods for ODEs

use super::{ExplicitRungeKutta, Ordinary, DormandPrince};
use crate::{
    Error, Status,
    alias::Evals,
    methods::h_init::InitialStepSize,
    interpolate::Interpolation,
    ode::{ODENumericalMethod, ODE},
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> ODENumericalMethod<T, V, D> for ExplicitRungeKutta<Ordinary, DormandPrince, T, V, D, O, S, I> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &V) -> Result<Evals, Error<T, V>>
    where
        F: ODE<T, V, D>,
    {
        let mut evals = Evals::new();

        // If h0 is zero, calculate initial step size
        if self.h0 == T::zero() {
            // Use adaptive step size calculation for Dormand-Prince methods
            self.h0 = InitialStepSize::<Ordinary>::compute(ode, t0, tf, y0, self.order, self.rtol, self.atol, self.h_min, self.h_max, &mut evals);
        }

        // Check bounds
        match validate_step_size_parameters::<T, V, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        // Initialize Statistics
        self.stiffness_counter = 0;

        // Initialize State
        self.t = t0;
        self.y = y0.clone();
        ode.diff(t0, y0, &mut self.k[0]);
        self.dydt = self.k[0];
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

        // Compute stages
        let mut y_stage = V::zeros();
        for i in 1..self.stages {
            y_stage = V::zeros();

            for j in 0..i {
                y_stage += self.k[j] * self.a[i][j];
            }
            y_stage = self.y + y_stage * self.h;

            ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);        
        }

        // The last stage will be used for stiffness detection
        let ysti = y_stage;

        // Calculate the line segment for the new y value
        let mut yseg = V::zeros();
        for i in 0..self.stages {
            yseg += self.k[i] * self.b[i];
        }

        // Calculate the new y value using the line segment
        let y_new = self.y + yseg * self.h;

        // Evaluate derivative at new point for error estimation
        let t_new = self.t + self.h;

        // Number of function evaluations
        evals.fcn += self.stages - 1; // We already have k[0]

        // Error Estimation
        let mut err = T::zero();
        let mut err2 = T::zero();
        let er = self.er.unwrap();
        let n = self.y.len();
        for i in 0..n {
            let sk = self.atol + self.rtol * self.y.get(i).abs().max(y_new.get(i).abs());
            if let Some(bh) = &self.bh {
                let mut erri = yseg.get(i);
                for j in 0..self.stages {
                    erri -= bh[j] * self.k[j].get(i);
                }
                err2 += (erri / sk).powi(2);
            }
            let mut erri = T::zero();
            for j in 0..self.stages {
                erri += er[j] * self.k[j].get(i);
            }
            err += (erri / sk).powi(2);
        }
        let mut deno = err + T::from_f64(0.01).unwrap() * err2;
        if deno <= T::zero() {
            deno = T::one();
        }
        err = self.h.abs() * err * (T::one() / (deno * T::from_usize(n).unwrap())).sqrt();

        // Step size settings
        let beta =  T::from_f64(0.0).unwrap();
        let expo1 = T::from_f64(1.0 / 8.0).unwrap();
        let facc1 = T::from_f64(1.0 / 0.33).unwrap();
        let facc2 = T::from_f64(1.0 / 6.0).unwrap();
        let facold = T::from_f64(1.0e-4).unwrap();
        // Computation of h_new
        let fac11 = err.powf(expo1);
        // Requirement that fac1 <= h_new/h <= fac2
        let fac = facc2.max(
            facc1.min(
                // Lund-stabilization
                (fac11 / facold.powf(beta)) / self.safety_factor
            )
        );
        let mut h_new = self.h / fac;

        // Determine if step is accepted
        if err <= T::one() {
            // Calculate the new derivative at the new point
            ode.diff(t_new, &y_new, &mut self.dydt);
            evals.fcn += 1;

            // stiffness detection
            let n_stiff_threshold = 100;
            if self.steps % n_stiff_threshold == 0 {
                let mut stdnum = T::zero();
                let mut stden = T::zero();
                let sqr = yseg - self.k[S-1];
                for i in 0..sqr.len() {
                    stdnum += sqr.get(i).powi(2);
                }
                let sqr = self.dydt - ysti;
                for i in 0..sqr.len() {
                    stden += sqr.get(i).powi(2);
                }

                if stden > T::zero() {
                    let h_lamb = self.h * (stdnum / stden).sqrt();
                    if h_lamb > T::from_f64(6.1).unwrap() {
                        self.non_stiffness_counter = 0;
                        self.stiffness_counter += 1;
                        if self.stiffness_counter == 15 {
                            // Early Exit Stiffness Detected
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
                } else {
                    self.non_stiffness_counter += 1;
                    if self.non_stiffness_counter == 6 {
                        self.stiffness_counter = 0;
                    }
                }
            }

            // Preparation for dense output / interpolation
            self.cont[0] = self.y;
            let ydiff = y_new - self.y;
            self.cont[1] = ydiff;
            let bspl = self.k[0] * self.h - ydiff;
            self.cont[2] = bspl;
            self.cont[3] = ydiff - self.dydt * self.h - bspl;

            // If method has dense output stages, compute them
            if let Some(bi) = &self.bi {
                // Compute extra stages for dense output
                if I > S {
                    // First dense output coefficient, k{i=order+1}, is the derivative at the new point
                    self.k[self.stages] = self.dydt;

                    for i in S+1..I {
                        let mut y_stage = V::zeros();
                        for j in 0..i {
                            y_stage += self.k[j] * self.a[i][j];
                        }
                        y_stage = self.y + y_stage * self.h;

                        ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
                        evals.fcn += 1;
                    }
                }

                // Compute dense output coefficients
                for i in 4..self.order {
                    self.cont[i] = V::zeros();
                    for j in 0..self.dense_stages {
                        self.cont[i] += self.k[j] * bi[i][j];
                    }
                    self.cont[i] = self.cont[i] * self.h;
                }
            }

            // For interpolation
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.k[0];
            self.h_prev = self.h;

            // Update the state with new values
            self.t = t_new;
            self.y = y_new;
            self.k[0] = self.dydt;

            // Check if previous step is rejected
            if let Status::RejectedStep = self.status {
                h_new = self.h.min(h_new);
                self.status = Status::Solving;
            }
        } else {
            // Step Rejected
            h_new = self.h / facc1.min(fac11 / self.safety_factor);
            self.status = Status::RejectedStep;
        }

        // Ensure step size is within bounds of h_min and h_max
        self.h = constrain_step_size(h_new, self.h_min, self.h_max);
        
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

impl<T: Real, V: State<T>, D: CallBackData, const O: usize, const S: usize, const I: usize> Interpolation<T, V> for ExplicitRungeKutta<Ordinary, DormandPrince, T, V, D, O, S, I> {
    fn interpolate(&mut self, t_interp: T) -> Result<V, Error<T, V>> {        // Check if interpolation is out of bounds
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
        // Single loop implementation of: cont[0] + (cont[1] + (cont[2] + (cont[3] + conpar*s1)*s)*s1)*s
        
        let mut y_interp = self.cont[0];
        
        // Build the entire nested polynomial in one loop using Horner's method
        let mut poly = self.cont[O - 1];
        
        // Work backwards from the highest coefficient to cont[1]
        for i in (1..O - 1).rev() {
            // Determine the multiplication factor based on the nesting pattern
            let factor = if i >= 4 {
                // For the higher-order part (conpar), alternate s and s1
                if (O - 1 - i) % 2 == 1 { s1 } else { s }
            } else {
                // For the main polynomial part, follow the specific pattern
                match i {
                    3 => s1,  // cont[3] gets multiplied by s1
                    2 => s,   // cont[2] gets multiplied by s  
                    1 => s1,  // cont[1] gets multiplied by s1
                    _ => s    // fallback
                }
            };
            
            poly = poly * factor + self.cont[i];
        }
        
        // Final multiplication by s for the outermost level
        y_interp = y_interp + poly * s;

        Ok(y_interp)
    }
}
