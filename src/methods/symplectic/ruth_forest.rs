//! Ruth-Forest 4th Order Symplectic Integrator

use crate::{
    error::Error,
    methods::{Fixed, Ordinary},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::validate_step_size_parameters,
};

use super::SymplecticIntegrator;

impl<T: Real, Y: State<T>> SymplecticIntegrator<Ordinary, Fixed, T, Y, 4> {
    /// Ruth-Forest (4th Order Symplectic Integrator).
    ///
    /// Ideal for conservative physics and orbital mechanics where
    /// energy conservation over long periods is required.
    ///
    /// # Arguments
    /// * `h` - Fixed step size.
    pub fn ruth_forest(h: T) -> Self {
        let two = T::from_f64(2.0).unwrap();
        let one = T::one();
        let two_pow_third = T::from_f64(1.259_921_049_894_873_2).unwrap(); // 2^(1/3)
        let theta = one / (two - two_pow_third);

        let c = [
            theta / two,
            (one - theta) / two,
            (one - theta) / two,
            theta / two,
        ];

        let d = [theta, one - two * theta, theta, T::zero()];

        Self {
            h,
            c,
            d,
            order: 4,
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> OrdinaryNumericalMethod<T, Y>
    for SymplecticIntegrator<Ordinary, Fixed, T, Y, 4>
{
    fn init<Eq>(&mut self, _ode: &Eq, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        Eq: ODE<T, Y> + ?Sized,
    {
        let evals = Evals::new();

        if y0.len() == 0 || y0.len() % 2 != 0 {
            return Err(Error::BadInput {
                msg: "Symplectic integrators require a non-empty state vector with an even number of elements (split equally between positions and momenta).".to_string(),
            });
        }

        if self.h == T::zero() {
            let duration = (tf - t0).abs();
            let default_steps = T::from_usize(100).unwrap();
            self.h = duration / default_steps;
        }

        match validate_step_size_parameters::<T, Y>(self.h, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => self.h = h0,
            Err(status) => return Err(status),
        }

        self.t = t0;
        self.y = y0.clone();
        self.y_prev = y0.clone();

        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<Eq>(&mut self, ode: &Eq) -> Result<Evals, Error<T, Y>>
    where
        Eq: ODE<T, Y> + ?Sized,
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

        self.y_prev = self.y.clone();

        let n = self.y.len();
        let half_n = n / 2;

        let mut y_next = self.y.clone();
        let mut f = self.y.zeros_like();

        for i in 0..4 {
            // Update position (q) using c_i and current momentum (p)
            ode.diff(self.t, &y_next, &mut f);
            evals.function += 1;
            for j in 0..half_n {
                y_next.set_component(
                    j,
                    y_next.get_component(j) + self.h * self.c[i] * f.get_component(j),
                );
            }

            // Update momentum (p) using d_i and current position (q)
            ode.diff(self.t, &y_next, &mut f);
            evals.function += 1;
            for j in half_n..n {
                y_next.set_component(
                    j,
                    y_next.get_component(j) + self.h * self.d[i] * f.get_component(j),
                );
            }
        }

        self.t += self.h;
        self.y = y_next;
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
        self.t - self.h
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

impl<T: Real, Y: State<T>> crate::interpolate::Interpolation<T, Y>
    for SymplecticIntegrator<Ordinary, Fixed, T, Y, 4>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        if t_interp < self.t_prev() || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev(),
                t_curr: self.t,
            });
        }

        let s = (t_interp - self.t_prev()) / self.h;
        let mut y_interp = self.y_prev.clone();
        for i in 0..y_interp.len() {
            let diff = self.y.get_component(i) - self.y_prev.get_component(i);
            y_interp.set_component(i, y_interp.get_component(i) + s * diff);
        }
        Ok(y_interp)
    }
}
