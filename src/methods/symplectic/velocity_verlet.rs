//! Velocity Verlet Symplectic Integrator

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

impl<T: Real, Y: State<T>> SymplecticIntegrator<Ordinary, Fixed, T, Y, 2> {
    /// Velocity Verlet (2nd Order Symplectic Integrator).
    ///
    /// Ideal for conservative physics and orbital mechanics where
    /// energy conservation over long periods is required.
    ///
    /// Formulated as a partitioned method with coefficients:
    /// c_1 = 0, c_2 = 1
    /// d_1 = 1/2, d_2 = 1/2
    ///
    /// # Arguments
    /// * `h` - Fixed step size.
    pub fn velocity_verlet(h: T) -> Self {
        let half = T::from_f64(0.5).unwrap();
        let c = [T::zero(), T::one()];
        let d = [half, half];
        Self {
            h,
            c,
            d,
            order: 2,
            ..Default::default()
        }
    }
}

// Note: To use symplectic integrators, we assume `y` is structured as `[q, p]` where
// `q` has length N/2 and `p` has length N/2. The ODE diff must calculate the full
// derivative. We will update `p` using `q`, and `q` using `p`.

impl<T: Real, Y: State<T>> OrdinaryNumericalMethod<T, Y>
    for SymplecticIntegrator<Ordinary, Fixed, T, Y, 2>
{
    fn init<Eq>(&mut self, _ode: &Eq, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        Eq: ODE<T, Y>,
    {
        let evals = Evals::new();

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

        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<Eq>(&mut self, ode: &Eq) -> Result<Evals, Error<T, Y>>
    where
        Eq: ODE<T, Y>,
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

        let n = self.y.len();
        let half_n = n / 2;

        let mut f = self.y.zeros_like();

        // Standard Velocity Verlet approach:
        // p_{n+1/2} = p_n + h/2 * f(q_n)
        // q_{n+1} = q_n + h * p_{n+1/2}
        // p_{n+1} = p_{n+1/2} + h/2 * f(q_{n+1})

        // Evaluate f(q_n). (Note: we evaluate the full system, but we only need the momentum derivative part).
        ode.diff(self.t, &self.y, &mut f);
        evals.function += 1;

        let mut y_next = self.y.clone();

        // 1. Update p (momentum) by half step: p_{n+1/2} = p_n + h/2 * f(q_n)
        for i in half_n..n {
            y_next.set_component(
                i,
                y_next.get_component(i) + self.h * self.d[0] * f.get_component(i),
            );
        }

        // 2. Update q (position) by full step: q_{n+1} = q_n + h * p_{n+1/2} (assuming unit mass, v = p)
        ode.diff(self.t, &y_next, &mut f);
        evals.function += 1;

        for i in 0..half_n {
            y_next.set_component(
                i,
                y_next.get_component(i) + self.h * self.c[1] * f.get_component(i),
            );
        }

        // 3. Update p (momentum) by another half step: p_{n+1} = p_{n+1/2} + h/2 * f(q_{n+1})
        ode.diff(self.t, &y_next, &mut f);
        evals.function += 1;

        for i in half_n..n {
            y_next.set_component(
                i,
                y_next.get_component(i) + self.h * self.d[1] * f.get_component(i),
            );
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
        &self.y
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
    for SymplecticIntegrator<Ordinary, Fixed, T, Y, 2>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Basic linear interpolation for symplectic integrators since we don't store dydt_prev
        if t_interp < self.t_prev() || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev(),
                t_curr: self.t,
            });
        }

        // This is a placeholder since we don't have dense output for these
        // methods currently. For precise continuous output of symplectic methods,
        // specialized interpolation is usually required.
        Ok(self.y.clone())
    }
}
