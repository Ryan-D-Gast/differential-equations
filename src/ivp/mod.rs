//! Unified builder for initial value problems.
//!
//! The high-level API owns the numerical method and output handler. Solvers mutate
//! their internal interpolation/history state during integration, so ownership makes
//! each `Ivp::solve` call single-use and avoids accidental solver reuse after a run.
//! Call the low-level `solve_*` functions directly when reference-based control is
//! required.

use crate::{
    dae::{AlgebraicNumericalMethod, DAE, solve_dae},
    dde::{DDE, DelayNumericalMethod, solve_dde},
    error::Error,
    interpolate::Interpolation,
    linalg::Matrix,
    methods::ToleranceConfig,
    ode::{
        AdjointCost, AdjointSolution, ForwardSensitivityODE, ODE, OrdinaryNumericalMethod,
        VaryParameters, solve_adjoint_sensitivity, solve_ode,
    },
    sde::{SDE, StochasticNumericalMethod, solve_sde},
    solout::{
        CrossingDirection, CrossingSolout, DefaultSolout, DenseSolout, EvenSolout, Event,
        EventWrappedSolout, HyperplaneCrossingSolout, Solout, TEvalSolout,
    },
    solution::Solution,
    tolerance::Tolerance,
    traits::{Real, State},
};
use nalgebra::DVector;

/// Unified builder for initial value problems (IVPs).
///
/// Consolidates solver configurations, output configurations, and events.
#[derive(Clone, Debug)]
pub struct Ivp<EqType, T: Real, Y: State<T>, Method, SoloutType> {
    equation: EqType,
    t0: T,
    tf: T,
    y0: Y,
    method: Method,
    solout: SoloutType,
}

/// Builder for forward sensitivity analysis of an ODE IVP.
#[derive(Clone, Debug)]
pub struct ForwardSensitivityIvp<Previous, T: Real, SoloutType = DefaultSolout> {
    previous: Previous,
    initial_sensitivity: Option<Matrix<T>>,
    solout: SoloutType,
    _marker: std::marker::PhantomData<T>,
}

/// Marker for using the forward method as the adjoint backward method.
#[derive(Clone, Copy, Debug)]
pub struct SameMethod;

/// Marker for an explicitly configured adjoint backward method.
#[derive(Clone, Debug)]
pub struct UseBackwardMethod<Method> {
    method: Method,
}

/// Builder for adjoint sensitivity analysis of an ODE IVP.
#[derive(Clone, Debug)]
pub struct AdjointSensitivityIvp<'c, Previous, Cost, BackwardMethod = SameMethod> {
    previous: Previous,
    cost: &'c Cost,
    backward_method: BackwardMethod,
    _marker: std::marker::PhantomData<()>,
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

impl<'a, F, T: Real, Y: State<T>> Ivp<OdeEq<'a, F>, T, Y, (), DefaultSolout> {
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

impl<'a, F, T: Real, Y: State<T>> Ivp<DaeEq<'a, F>, T, Y, (), DefaultSolout> {
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

impl<'a, F, T: Real, Y: State<T>> Ivp<SdeEq<'a, F>, T, Y, (), DefaultSolout> {
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
    Ivp<DdeEq<'a, L, F, H>, T, Y, (), DefaultSolout>
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

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> Ivp<EqType, T, Y, Method, SoloutType> {
    fn with_method<NextMethod>(
        self,
        method: NextMethod,
    ) -> Ivp<EqType, T, Y, NextMethod, SoloutType> {
        Ivp {
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
    ) -> Ivp<EqType, T, Y, NextMethod, SoloutType> {
        Ivp {
            equation: self.equation,
            t0: self.t0,
            tf: self.tf,
            y0: self.y0,
            method: map(self.method),
            solout: self.solout,
        }
    }

    fn with_solout<NextSolout>(self, solout: NextSolout) -> Ivp<EqType, T, Y, Method, NextSolout> {
        Ivp {
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
    pub fn method<SNew>(self, method: SNew) -> Ivp<EqType, T, Y, SNew, SoloutType> {
        self.with_method(method)
    }

    /// Set a custom solout function.
    pub fn solout<ONew>(self, solout: ONew) -> Ivp<EqType, T, Y, Method, ONew> {
        self.with_solout(solout)
    }

    /// Output evenly spaced points between the initial and final time.
    /// Note that this does not include the solution of the calculated steps.
    pub fn even(self, dt: T) -> Ivp<EqType, T, Y, Method, EvenSolout<T>> {
        let solout = EvenSolout::new(dt, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Use the Dense Output method to output n number of interpolation points between each step.
    /// Note this includes the solution of the calculated steps.
    pub fn dense(self, n: usize) -> Ivp<EqType, T, Y, Method, DenseSolout> {
        self.with_solout(DenseSolout::new(n))
    }

    /// Use the provided time points for evaluation instead of the default method.
    /// Note this does not include the solution of the calculated steps.
    pub fn t_eval(self, points: impl AsRef<[T]>) -> Ivp<EqType, T, Y, Method, TEvalSolout<T>> {
        let solout = TEvalSolout::new(points, self.t0, self.tf);
        self.with_solout(solout)
    }

    /// Wrap current solout with event detection while preserving original output strategy.
    pub fn event<'a, E>(
        self,
        event: &'a E,
    ) -> Ivp<EqType, T, Y, Method, EventWrappedSolout<'a, T, Y, SoloutType, E>>
    where
        E: Event<T, Y>,
        SoloutType: Solout<T, Y>,
    {
        Ivp {
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
    ) -> Ivp<EqType, T, Y, Method, CrossingSolout<T>> {
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
    ) -> Ivp<EqType, T, Y, Method, HyperplaneCrossingSolout<T, Y1, Y>> {
        let solout =
            HyperplaneCrossingSolout::new(point, normal, extractor).with_direction(direction);
        self.with_solout(solout)
    }
}

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<OdeEq<'a, F>, T, Y, Method, SoloutType> {
    /// Activate forward sensitivity analysis for this ODE IVP.
    ///
    /// The builder constructs the augmented state `[y0, S0]` dynamically. By
    /// default `S0` is zero, which is appropriate when the initial condition
    /// does not depend on parameters. Use
    /// [`ForwardSensitivityIvp::initial_sensitivity`] when `dy0/dp` is known.
    pub fn forward_sensitivity(self) -> ForwardSensitivityIvp<Self, T> {
        ForwardSensitivityIvp {
            previous: self,
            initial_sensitivity: None,
            solout: DefaultSolout::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Activate adjoint sensitivity analysis for this ODE IVP.
    ///
    /// By default, the forward method is cloned and reused for the backward
    /// adjoint solve. Use [`AdjointSensitivityIvp::backward_method`] to provide
    /// a distinct method.
    pub fn adjoint_sensitivity<'c, Cost>(
        self,
        cost: &'c Cost,
    ) -> AdjointSensitivityIvp<'c, Self, Cost> {
        AdjointSensitivityIvp {
            previous: self,
            cost,
            backward_method: SameMethod,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Previous, T: Real, SoloutType> ForwardSensitivityIvp<Previous, T, SoloutType> {
    /// Set the initial sensitivity matrix `S0 = dy0/dp`.
    ///
    /// The matrix must have shape `y0.len() x params.len()`.
    pub fn initial_sensitivity(mut self, sensitivity: Matrix<T>) -> Self {
        self.initial_sensitivity = Some(sensitivity);
        self
    }

    /// Set a custom output strategy for the augmented forward sensitivity solve.
    pub fn solout<NextSolout>(
        self,
        solout: NextSolout,
    ) -> ForwardSensitivityIvp<Previous, T, NextSolout> {
        ForwardSensitivityIvp {
            previous: self.previous,
            initial_sensitivity: self.initial_sensitivity,
            solout,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Previous, T: Real, SoloutType> ForwardSensitivityIvp<Previous, T, SoloutType> {
    /// Use dense output for the augmented forward sensitivity solve.
    pub fn dense(self, n: usize) -> ForwardSensitivityIvp<Previous, T, DenseSolout> {
        self.solout(DenseSolout::new(n))
    }
}

impl<'a, F, T, Y, Method, BaseSolout, SensSolout>
    ForwardSensitivityIvp<Ivp<OdeEq<'a, F>, T, Y, Method, BaseSolout>, T, SensSolout>
where
    T: Real,
    Y: State<T>,
    F: VaryParameters<T, Y>,
    Method: OrdinaryNumericalMethod<T, DVector<T>> + Interpolation<T, DVector<T>>,
    SensSolout: Solout<T, DVector<T>>,
{
    /// Solve the activated forward sensitivity IVP.
    pub fn solve(mut self) -> Result<Solution<T, DVector<T>>, Error<T, DVector<T>>> {
        let state_dim = self.previous.y0.len();
        let params = self.previous.equation.ode.parameters();
        let param_dim = params.len();
        let mut y_aug0 = DVector::zeros(state_dim + state_dim * param_dim);
        for i in 0..state_dim {
            y_aug0[i] = self.previous.y0.get(i);
        }

        if let Some(s0) = &self.initial_sensitivity {
            if s0.dims() != (state_dim, param_dim) {
                return Err(Error::BadInput {
                    msg: format!(
                        "Initial sensitivity must have shape {} x {}, got {} x {}.",
                        state_dim,
                        param_dim,
                        s0.nrows(),
                        s0.ncols()
                    ),
                });
            }
            for param_idx in 0..param_dim {
                let offset = state_dim + param_idx * state_dim;
                for state_idx in 0..state_dim {
                    y_aug0[offset + state_idx] = s0[(state_idx, param_idx)];
                }
            }
        }

        let sensitivity_ode =
            ForwardSensitivityODE::new(self.previous.equation.ode, self.previous.y0.clone());
        solve_ode(
            &mut self.previous.method,
            &sensitivity_ode,
            self.previous.t0,
            self.previous.tf,
            &y_aug0,
            &mut self.solout,
        )
    }
}

impl<'c, Previous, Cost, BackwardMethod> AdjointSensitivityIvp<'c, Previous, Cost, BackwardMethod> {
    /// Use a distinct numerical method for the backward adjoint solve.
    pub fn backward_method<Method>(
        self,
        method: Method,
    ) -> AdjointSensitivityIvp<'c, Previous, Cost, UseBackwardMethod<Method>> {
        AdjointSensitivityIvp {
            previous: self.previous,
            cost: self.cost,
            backward_method: UseBackwardMethod { method },
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'c, 'a, F, T, Y, Method, SoloutType, Cost>
    AdjointSensitivityIvp<'c, Ivp<OdeEq<'a, F>, T, Y, Method, SoloutType>, Cost, SameMethod>
where
    T: Real,
    Y: State<T>,
    F: VaryParameters<T, Y>,
    Cost: AdjointCost<T, Y, F>,
    Method: OrdinaryNumericalMethod<T, Y>
        + Interpolation<T, Y>
        + OrdinaryNumericalMethod<T, crate::ode::AdjointState<T, Y, F::Params>>
        + Interpolation<T, crate::ode::AdjointState<T, Y, F::Params>>
        + Clone,
{
    /// Solve the activated adjoint sensitivity IVP.
    pub fn solve(mut self) -> Result<AdjointSolution<T, Y, F::Params>, Error<T, Y>> {
        let mut backward_method = self.previous.method.clone();
        solve_adjoint_sensitivity(
            &mut self.previous.method,
            &mut backward_method,
            self.previous.equation.ode,
            self.cost,
            self.previous.t0,
            self.previous.tf,
            &self.previous.y0,
        )
    }
}

impl<'c, 'a, F, T, Y, Method, SoloutType, Cost, BackwardMethod>
    AdjointSensitivityIvp<
        'c,
        Ivp<OdeEq<'a, F>, T, Y, Method, SoloutType>,
        Cost,
        UseBackwardMethod<BackwardMethod>,
    >
where
    T: Real,
    Y: State<T>,
    F: VaryParameters<T, Y>,
    Cost: AdjointCost<T, Y, F>,
    Method: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>,
    BackwardMethod: OrdinaryNumericalMethod<T, crate::ode::AdjointState<T, Y, F::Params>>
        + Interpolation<T, crate::ode::AdjointState<T, Y, F::Params>>,
{
    /// Solve the activated adjoint sensitivity IVP.
    pub fn solve(mut self) -> Result<AdjointSolution<T, Y, F::Params>, Error<T, Y>> {
        solve_adjoint_sensitivity(
            &mut self.previous.method,
            &mut self.backward_method.method,
            self.previous.equation.ode,
            self.cost,
            self.previous.t0,
            self.previous.tf,
            &self.previous.y0,
        )
    }
}

impl<EqType, T: Real, Y: State<T>, Method, SoloutType> Ivp<EqType, T, Y, Method, SoloutType>
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

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<OdeEq<'a, F>, T, Y, Method, SoloutType>
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

impl<'a, F, T: Real, Y: State<T>, Method, SoloutType> Ivp<DaeEq<'a, F>, T, Y, Method, SoloutType>
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

impl<'a, F, T: Real, Y: State<T> + Copy, Method, SoloutType>
    Ivp<SdeEq<'a, F>, T, Y, Method, SoloutType>
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

impl<'a, const L: usize, F, H, T: Real, Y: State<T> + Copy, Method, SoloutType>
    Ivp<DdeEq<'a, L, F, H>, T, Y, Method, SoloutType>
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
