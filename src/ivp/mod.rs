//! Unified builder for initial value problems.
//!
//! The high-level API owns the numerical method and output handler. Solvers mutate
//! their internal interpolation/history state during integration, so ownership makes
//! each `IVP::solve` call single-use and avoids accidental solver reuse after a run.
//! Call the low-level `solve_*` functions directly when reference-based control is
//! required.

use crate::{
    dae::{AlgebraicNumericalMethod, DAE, solve_dae},
    dde::{DDE, DelayNumericalMethod, solve_dde},
    error::Error,
    interpolate::Interpolation,
    methods::ToleranceConfig,
    ode::{ODE, OrdinaryNumericalMethod, solve_ode},
    sde::{SDE, StochasticNumericalMethod, solve_sde},
    solout::{
        CrossingDirection, CrossingSolout, DefaultSolout, DenseSolout, EvenSolout, Event,
        EventWrappedSolout, HyperplaneCrossingSolout, Solout, TEvalSolout,
    },
    solution::Solution,
    tolerance::Tolerance,
    traits::{Real, State},
};

/// Unified builder for initial value problems (IVPs).
///
/// Consolidates solver configurations, output configurations, and events.
#[derive(Clone, Debug)]
pub struct IVP<EqType, T: Real, Y: State<T>, Method, SoloutType> {
    equation: EqType,
    t0: T,
    tf: T,
    y0: Y,
    method: Method,
    solout: SoloutType,
}

/// Marker for ordinary differential equations.
#[derive(Debug)]
pub struct OdeEq<'a, F> {
    ode: &'a F,
}

impl<F> Clone for OdeEq<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for OdeEq<'_, F> {}

/// Marker for differential algebraic equations.
#[derive(Debug)]
pub struct DaeEq<'a, F> {
    dae: &'a F,
}

impl<F> Clone for DaeEq<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for DaeEq<'_, F> {}

/// Marker for stochastic differential equations.
#[derive(Debug)]
pub struct SdeEq<'a, F> {
    sde: &'a mut F,
}

/// Marker for delay differential equations.
#[derive(Debug)]
pub struct DdeEq<'a, const L: usize, F, H> {
    dde: &'a F,
    history: H,
}

impl<const L: usize, F, H: Clone> Clone for DdeEq<'_, L, F, H> {
    fn clone(&self) -> Self {
        Self {
            dde: self.dde,
            history: self.history.clone(),
        }
    }
}

impl<'a, F, T: Real, Y: State<T>> IVP<OdeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new initial value problem for an ordinary differential equation.
    pub fn ode(system: &'a F, t0: T, tf: T, y0: Y) -> Self {
        Self {
            equation: OdeEq { ode: system },
            t0,
            tf,
            y0,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<'a, F, T: Real, Y: State<T>> IVP<DaeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new initial value problem for a differential algebraic equation.
    pub fn dae(system: &'a F, t0: T, tf: T, y0: Y) -> Self {
        Self {
            equation: DaeEq { dae: system },
            t0,
            tf,
            y0,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<'a, F, T: Real, Y: State<T>> IVP<SdeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new initial value problem for a stochastic differential equation.
    pub fn sde(system: &'a mut F, t0: T, tf: T, y0: Y) -> Self {
        Self {
            equation: SdeEq { sde: system },
            t0,
            tf,
            y0,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<'a, F, H, T: Real, Y: State<T>, const L: usize>
    IVP<DdeEq<'a, L, F, H>, T, Y, (), DefaultSolout>
{
    /// Create a new initial value problem for a delay differential equation.
    pub fn dde(system: &'a F, t0: T, tf: T, y0: Y, history_function: H) -> Self {
        Self {
            equation: DdeEq {
                dde: system,
                history: history_function,
            },
            t0,
            tf,
            y0,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> IVP<EqType, T, Y, Method, SoloutType> {
    fn with_method<NextMethod>(
        self,
        method: NextMethod,
    ) -> IVP<EqType, T, Y, NextMethod, SoloutType> {
        IVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method,
            solout: self.solout,
        }
    }

    fn map_method<NextMethod>(
        self,
        map: impl FnOnce(Method) -> NextMethod,
    ) -> IVP<EqType, T, Y, NextMethod, SoloutType> {
        IVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: map(self.method),
            solout: self.solout,
        }
    }

    fn with_solout<NextSolout>(self, solout: NextSolout) -> IVP<EqType, T, Y, Method, NextSolout> {
        IVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }

    /// Set the numerical method to be used.
    ///
    /// The builder owns the method because solving mutates method state. Construct
    /// a fresh method for each solve, or use the low-level `solve_*` functions when
    /// you need to manage a mutable solver reference directly.
    pub fn method<SNew>(self, method: SNew) -> IVP<EqType, T, Y, SNew, SoloutType> {
        self.with_method(method)
    }

    /// Set a custom solout function.
    pub fn solout<ONew>(self, solout: ONew) -> IVP<EqType, T, Y, Method, ONew> {
        self.with_solout(solout)
    }

    /// Output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    pub fn even(self, dt: T) -> IVP<EqType, T, Y, Method, EvenSolout<T>> {
        let solout = EvenSolout::new(dt, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Use the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    pub fn dense(self, n: usize) -> IVP<EqType, T, Y, Method, DenseSolout> {
        self.with_solout(DenseSolout::new(n))
    }

    /// Use the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    pub fn t_eval(self, points: impl AsRef<[T]>) -> IVP<EqType, T, Y, Method, TEvalSolout<T>> {
        let solout = TEvalSolout::new(points, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Wrap current solout with event detection while preserving original output strategy.
    pub fn event<'a, E>(
        self,
        event: &'a E,
    ) -> IVP<EqType, T, Y, Method, EventWrappedSolout<'a, T, Y, SoloutType, E>>
    where
        E: Event<T, Y>,
        SoloutType: Solout<T, Y>,
    {
        IVP {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout: EventWrappedSolout::new(self.solout, event, self.t0, self.tf),
        }
    }

    /// Uses the CrossingSolout method to output points when a specific component crosses a threshold.
    /// Note this does not include the solution of the calculated steps.
    pub fn crossing(
        self,
        component_idx: usize,
        threshold: T,
        direction: CrossingDirection,
    ) -> IVP<EqType, T, Y, Method, CrossingSolout<T>> {
        let crossing_solout =
            CrossingSolout::new(component_idx, threshold).with_direction(direction);
        self.with_solout(crossing_solout)
    }

    /// Uses the HyperplaneCrossingSolout method to output points when a specific hyperplane is crossed.
    /// Note this does not include the solution of the calculated steps.
    pub fn hyperplane_crossing<Y1: State<T>>(
        self,
        point: Y1,
        normal: Y1,
        extractor: fn(&Y) -> Y1,
        direction: CrossingDirection,
    ) -> IVP<EqType, T, Y, Method, HyperplaneCrossingSolout<T, Y1, Y>> {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);
        self.with_solout(solout)
    }
}

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> IVP<EqType, T, Y, Method, SoloutType>
where
    Method: ToleranceConfig<T>,
{
    /// Set relative tolerance on the underlying solver.
    pub fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.map_method(|method| method.rtol(rtol))
    }

    /// Set absolute tolerance on the underlying solver.
    pub fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.map_method(|method| method.atol(atol))
    }
}

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> IVP<OdeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: ODE<T, Y>,
    Method: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: Solout<T, Y>,
{
    /// Solve the ODE initial value problem.
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        solve_ode(
            &mut self.method,
            self.equation.ode,
            self.t0,
            self.tf,
            &self.y0,
            &mut self.solout,
        )
    }
}

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> IVP<DaeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: DAE<T, Y>,
    Method: AlgebraicNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: Solout<T, Y>,
{
    /// Solve the DAE initial value problem.
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        solve_dae(
            &mut self.method,
            self.equation.dae,
            self.t0,
            self.tf,
            &self.y0,
            &mut self.solout,
        )
    }
}

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> IVP<SdeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: SDE<T, Y>,
    Method: StochasticNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: Solout<T, Y>,
{
    /// Solve the SDE initial value problem.
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        solve_sde(
            &mut self.method,
            self.equation.sde,
            self.t0,
            self.tf,
            &self.y0,
            &mut self.solout,
        )
    }
}

impl<'a, const L: usize, F, H, T: Real, Y: State<T>, Method, SoloutType>
    IVP<DdeEq<'a, L, F, H>, T, Y, Method, SoloutType>
where
    F: DDE<L, T, Y>,
    H: Fn(T) -> Y + Clone,
    Method: DelayNumericalMethod<L, T, Y, H> + Interpolation<T, Y>,
    SoloutType: Solout<T, Y>,
{
    /// Solve the DDE initial value problem.
    pub fn solve(mut self) -> Result<Solution<T, Y>, Error<T, Y>> {
        solve_dde(
            &mut self.method,
            self.equation.dde,
            self.t0,
            self.tf,
            &self.y0,
            self.equation.history.clone(),
            &mut self.solout,
        )
    }
}
