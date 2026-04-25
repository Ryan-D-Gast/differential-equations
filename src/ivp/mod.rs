//! Unified Builder for Initial Value Problems

use crate::traits::{Real, State};
use crate::solout::DefaultSolout;

/// Unified builder for Initial Value Problems (IVPs).
/// Consolidates solver configurations, output configurations, and events.
#[derive(Clone, Debug)]
pub struct Ivp<EqType, T: Real, Y: State<T>, Method, Solout> {
    pub equation: EqType,
    pub t0: T,
    pub tf: T,
    pub y0: Y,
    pub method: Method,
    pub solout: Solout,
}

/// Marker struct for Ordinary Differential Equations
#[derive(Clone, Debug)]
pub struct OdeEq<'a, F> { pub ode: &'a F }

/// Marker struct for Differential Algebraic Equations
#[derive(Clone, Debug)]
pub struct DaeEq<'a, F> { pub dae: &'a F }

/// Marker struct for Stochastic Differential Equations
#[derive(Debug)]
pub struct SdeEq<'a, F> { pub sde: &'a mut F }

/// Marker struct for Delay Differential Equations
#[derive(Clone, Debug)]
pub struct DdeEq<'a, const L: usize, F, H> { pub dde: &'a F, pub history: H }

impl<'a, F, T: Real, Y: State<T>> Ivp<OdeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new Initial Value Problem for an Ordinary Differential Equation
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

impl<'a, F, T: Real, Y: State<T>> Ivp<DaeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new Initial Value Problem for a Differential Algebraic Equation
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

impl<'a, F, T: Real, Y: State<T>> Ivp<SdeEq<'a, F>, T, Y, (), DefaultSolout> {
    /// Create a new Initial Value Problem for a Stochastic Differential Equation
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

impl<'a, F, H, T: Real, Y: State<T>, const L: usize> Ivp<DdeEq<'a, L, F, H>, T, Y, (), DefaultSolout> {
    /// Create a new Initial Value Problem for a Delay Differential Equation
    pub fn dde(system: &'a F, t0: T, tf: T, y0: Y, history_function: H) -> Self {
        Self {
            equation: DdeEq { dde: system, history: history_function },
            t0,
            tf,
            y0,
            method: (),
            solout: DefaultSolout::new(),
        }
    }
}

use crate::solout::{DenseSolout, EvenSolout, TEvalSolout, EventWrappedSolout, Event};

impl<EqType, T: Real, Y: State<T>, Method, Solout> Ivp<EqType, T, Y, Method, Solout> {
    /// Set the numerical method to be used.
    pub fn method<SNew>(self, method: SNew) -> Ivp<EqType, T, Y, SNew, Solout> {
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method,
            solout: self.solout,
        }
    }

    /// Set a custom solout function.
    pub fn solout<ONew>(self, solout: ONew) -> Ivp<EqType, T, Y, Method, ONew> {
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }

    /// Output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    pub fn even(self, dt: T) -> Ivp<EqType, T, Y, Method, EvenSolout<T>> {
        let solout = EvenSolout::new(dt, self.t0, self.tf);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }

    /// Use the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    pub fn dense(self, n: usize) -> Ivp<EqType, T, Y, Method, DenseSolout> {
        let solout = DenseSolout::new(n);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }

    /// Use the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    pub fn t_eval(self, points: impl AsRef<[T]>) -> Ivp<EqType, T, Y, Method, TEvalSolout<T>> {
        let solout = TEvalSolout::new(points, self.t0, self.tf);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }

    /// Wrap current solout with event detection while preserving original output strategy.
    pub fn event<'a, E>(self, event: &'a E) -> Ivp<EqType, T, Y, Method, EventWrappedSolout<'a, T, Y, Solout, E>>
    where E: Event<T, Y>,
          Solout: crate::solout::Solout<T, Y>
    {
        let wrapped = EventWrappedSolout::new(self.solout, event, self.t0, self.tf);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout: wrapped,
        }
    }
}

use crate::solout::{CrossingSolout, HyperplaneCrossingSolout, CrossingDirection};

impl<EqType, T: Real, Y: State<T>, Method, Solout> Ivp<EqType, T, Y, Method, Solout> {
    /// Uses the CrossingSolout method to output points when a specific component crosses a threshold.
    /// Note this does not include the solution of the calculated steps.
    pub fn crossing(self, component_idx: usize, threshhold: T, direction: CrossingDirection) -> Ivp<EqType, T, Y, Method, CrossingSolout<T>> {
        let crossing_solout = CrossingSolout::new(component_idx, threshhold).with_direction(direction);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout: crossing_solout,
        }
    }

    /// Uses the HyperplaneCrossingSolout method to output points when a specific hyperplane is crossed.
    /// Note this does not include the solution of the calculated steps.
    pub fn hyperplane_crossing<Y1: State<T>>(self, point: Y1, normal: Y1, extractor: fn(&Y) -> Y1, direction: CrossingDirection) -> Ivp<EqType, T, Y, Method, HyperplaneCrossingSolout<T, Y1, Y>> {
        let solout = HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method,
            solout,
        }
    }
}

use crate::methods::ToleranceConfig;
use crate::tolerance::Tolerance;

impl<EqType, T: Real, Y: State<T>, Method, Solout> Ivp<EqType, T, Y, Method, Solout>
where Method: ToleranceConfig<T> {
    /// Set relative tolerance on the underlying solver.
    pub fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method.rtol(rtol),
            solout: self.solout,
        }
    }

    /// Set absolute tolerance on the underlying solver.
    pub fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: self.method.atol(atol),
            solout: self.solout,
        }
    }
}

use crate::error::Error;
use crate::solution::Solution;
use crate::interpolate::Interpolation;

// Implement solve for ODE
use crate::ode::{ODE, OrdinaryNumericalMethod, solve_ode};

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<OdeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: ODE<T, Y>,
    Method: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: crate::solout::Solout<T, Y>,
{
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

// Implement solve for DAE
use crate::dae::{DAE, AlgebraicNumericalMethod, solve_dae};

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<DaeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: DAE<T, Y>,
    Method: AlgebraicNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: crate::solout::Solout<T, Y>,
{
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

// Implement solve for SDE
use crate::sde::{SDE, StochasticNumericalMethod, solve_sde};

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<SdeEq<'a, F>, T, Y, Method, SoloutType>
where
    F: SDE<T, Y>,
    Method: StochasticNumericalMethod<T, Y> + Interpolation<T, Y>,
    SoloutType: crate::solout::Solout<T, Y>,
{
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

// Implement solve for DDE
use crate::dde::{DDE, DelayNumericalMethod, solve_dde};

impl<'a, const L: usize, F, H, T: Real, Y: State<T>, Method, SoloutType> Ivp<DdeEq<'a, L, F, H>, T, Y, Method, SoloutType>
where
    F: DDE<L, T, Y>,
    H: Fn(T) -> Y + Clone,
    Method: DelayNumericalMethod<L, T, Y, H> + Interpolation<T, Y>,
    SoloutType: crate::solout::Solout<T, Y>,
{
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
