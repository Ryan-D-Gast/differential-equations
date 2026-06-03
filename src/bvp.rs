//! Boundary value problem builder API.

use crate::{
    error::Error,
    methods::{ToleranceConfig, bvp::BVPMethod},
    ode::{ODE, solve_bvp},
    solout::{
        CrossingDirection, CrossingSolout, DefaultSolout, DenseSolout, EvenSolout, Event,
        EventWrappedSolout, HyperplaneCrossingSolout, Solout, TEvalSolout,
    },
    solution::Solution,
    tolerance::Tolerance,
    traits::{DefaultState, Real, State},
};

/// Builder for boundary value problems.
///
/// Use [`BVP::ode`] for an ODE system that implements [`ODE`] and [`Boundary`],
/// or [`BVP::ode_from_fn`] when closures are more convenient.
#[derive(Clone, Debug)]
pub struct BVP<EqType, T: Real, Y: State<T>, Method, SoloutType> {
    equation: EqType,
    t0: T,
    tf: T,
    y_guess: Y,
    method: Method,
    solout: SoloutType,
}

/// Boundary condition residual for a boundary value problem.
///
/// The solver searches for an initial state such that this residual is zero.
pub trait Boundary<T = f64, Y = DefaultState<T>>
where
    T: Real,
    Y: State<T>,
{
    /// Compute the boundary residual `res = g(y_a, y_b)`.
    fn boundary(&self, y_a: &Y, y_b: &Y, res: &mut Y);
}

impl<EqType, T: Real, Y: State<T>> Boundary<T, Y> for &EqType
where
    EqType: Boundary<T, Y> + ?Sized,
{
    fn boundary(&self, y_a: &Y, y_b: &Y, res: &mut Y) {
        (*self).boundary(y_a, y_b, res);
    }
}

/// Closure-backed ODE boundary value problem.
#[derive(Clone, Debug)]
pub struct OdeBvpFromFn<F, B> {
    diff: F,
    boundary: B,
}

impl<F, B> OdeBvpFromFn<F, B> {
    /// Create a closure-backed ODE BVP definition.
    pub fn new(diff: F, boundary: B) -> Self {
        Self { diff, boundary }
    }
}

impl<F, B, T, Y> ODE<T, Y> for OdeBvpFromFn<F, B>
where
    T: Real,
    Y: State<T>,
    F: Fn(T, &Y, &mut Y),
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        (self.diff)(t, y, dydt);
    }
}

impl<F, B, T, Y> Boundary<T, Y> for OdeBvpFromFn<F, B>
where
    T: Real,
    Y: State<T>,
    F: Fn(T, &Y, &mut Y),
    B: Fn(&Y, &Y, &mut Y),
{
    fn boundary(&self, y_a: &Y, y_b: &Y, res: &mut Y) {
        (self.boundary)(y_a, y_b, res);
    }
}

impl<'a, F, T, Y> BVP<&'a F, T, Y, (), DefaultSolout>
where
    T: Real,
    Y: State<T>,
    F: ODE<T, Y> + Boundary<T, Y> + ?Sized,
{
    /// Create a boundary value problem for an ODE.
    pub fn ode(system: &'a F, t0: T, tf: T, y_guess: Y) -> Self {
        Self {
            equation: system,
            t0,
            tf,
            y_guess,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<F, B, T, Y> BVP<OdeBvpFromFn<F, B>, T, Y, (), DefaultSolout>
where
    T: Real,
    Y: State<T>,
    F: Fn(T, &Y, &mut Y),
    B: Fn(&Y, &Y, &mut Y),
{
    /// Create a boundary value problem for an ODE from closures.
    pub fn ode_from_fn(diff: F, boundary: B, t0: T, tf: T, y_guess: Y) -> Self {
        Self {
            equation: OdeBvpFromFn::new(diff, boundary),
            t0,
            tf,
            y_guess,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> BVP<EqType, T, Y, Method, SoloutType> {
    fn with_method<NextMethod>(
        self,
        method: NextMethod,
    ) -> BVP<EqType, T, Y, NextMethod, SoloutType> {
        BVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method,
            solout: self.solout,
        }
    }

    fn with_solout<NextSolout>(self, solout: NextSolout) -> BVP<EqType, T, Y, Method, NextSolout> {
        BVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method: self.method,
            solout,
        }
    }

    /// Set the numerical method used to solve the BVP.
    pub fn method<NextMethod>(
        self,
        method: NextMethod,
    ) -> BVP<EqType, T, Y, NextMethod, SoloutType> {
        self.with_method(method)
    }

    /// Set a custom solout function for the final converged trajectory.
    pub fn solout<NextSolout>(self, solout: NextSolout) -> BVP<EqType, T, Y, Method, NextSolout> {
        self.with_solout(solout)
    }

    /// Output evenly spaced points between the initial and final time.
    ///
    /// This controls the output of the final converged IVP trajectory.
    pub fn even(self, dt: T) -> BVP<EqType, T, Y, Method, EvenSolout<T>> {
        let solout = EvenSolout::new(dt, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Output dense interpolation points between accepted solver steps.
    ///
    /// This controls the output of the final converged IVP trajectory.
    pub fn dense(self, n: usize) -> BVP<EqType, T, Y, Method, DenseSolout> {
        self.with_solout(DenseSolout::new(n))
    }

    /// Use provided time points for final trajectory output.
    pub fn t_eval(self, points: impl AsRef<[T]>) -> BVP<EqType, T, Y, Method, TEvalSolout<T>> {
        let solout = TEvalSolout::new(points, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Wrap current final-trajectory output with event detection.
    pub fn event<'a, E>(
        self,
        event: &'a E,
    ) -> BVP<EqType, T, Y, Method, EventWrappedSolout<'a, T, Y, SoloutType, E>>
    where
        E: Event<T, Y>,
        SoloutType: Solout<T, Y>,
    {
        BVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method: self.method,
            solout: EventWrappedSolout::new(self.solout, event, self.t0, self.tf),
        }
    }

    /// Output points where a component crosses a threshold on the final trajectory.
    pub fn crossing(
        self,
        component_idx: usize,
        threshold: T,
        direction: CrossingDirection,
    ) -> BVP<EqType, T, Y, Method, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshold).with_direction(direction);
        self.with_solout(crossing_solout)
    }

    /// Output points where a projected final trajectory crosses a hyperplane.
    pub fn hyperplane_crossing<Y1: State<T>>(
        self,
        point: Y1,
        normal: Y1,
        extractor: fn(&Y) -> Y1,
        direction: CrossingDirection,
    ) -> BVP<EqType, T, Y, Method, HyperplaneCrossingSolout<T, Y1, Y>> {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);
        self.with_solout(solout)
    }
}

impl<EqType, T, Y, Method, SoloutType> BVP<EqType, T, Y, Method, SoloutType>
where
    T: Real,
    Y: State<T>,
    EqType: ODE<T, Y> + Boundary<T, Y>,
    Method: BVPMethod<T, Y>,
    SoloutType: Solout<T, Y>,
{
    /// Solve the boundary value problem.
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        solve_bvp(
            &mut self.method,
            &self.equation,
            self.t0,
            self.tf,
            &self.y_guess,
            &mut self.solout,
        )
    }
}

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> BVP<EqType, T, Y, Method, SoloutType>
where
    Method: ToleranceConfig<T>,
{
    /// Set relative tolerance on the underlying solver.
    pub fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        BVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method: self.method.rtol(rtol),
            solout: self.solout,
        }
    }

    /// Set absolute tolerance on the underlying solver.
    pub fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        BVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y_guess: self.y_guess,
            method: self.method.atol(atol),
            solout: self.solout,
        }
    }
}
