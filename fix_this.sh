cat << 'INNER_EOF' > src/ivp/mod.rs
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
INNER_EOF

# Add mod ivp
awk '
/pub mod ode;/ { print "pub mod ivp;"; print $0; next }
{ print $0 }
' src/lib.rs > src/lib_tmp.rs
mv src/lib_tmp.rs src/lib.rs

# patch ToleranceConfig
cat << 'TRAIT' >> src/methods/mod.rs

use crate::tolerance::Tolerance;
use crate::traits::Real;

/// Trait to allow configuring tolerances on numerical methods generically.
pub trait ToleranceConfig<T: Real> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self;
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self;
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T> for crate::methods::ExplicitRungeKutta<E, F, T, Y, O, S, I> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T> for crate::methods::ImplicitRungeKutta<E, F, T, Y, O, S, I> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}

impl<E, F, T: Real, Y: crate::traits::State<T>, const O: usize, const S: usize, const I: usize> ToleranceConfig<T> for crate::methods::DiagonallyImplicitRungeKutta<E, F, T, Y, O, S, I> {
    fn rtol<V: Into<Tolerance<T>>>(self, rtol: V) -> Self {
        self.rtol(rtol)
    }
    fn atol<V: Into<Tolerance<T>>>(self, atol: V) -> Self {
        self.atol(atol)
    }
}
TRAIT

sed -i '/mod problem;/d' src/ode/mod.rs
sed -i '/pub use problem::ODEProblem;/d' src/ode/mod.rs
sed -i '/mod problem;/d' src/dae/mod.rs
sed -i '/pub use problem::DAEProblem;/d' src/dae/mod.rs
sed -i '/mod problem;/d' src/sde/mod.rs
sed -i '/pub use problem::SDEProblem;/d' src/sde/mod.rs
sed -i '/mod problem;/d' src/dde/mod.rs
sed -i '/pub use problem::DDEProblem;/d' src/dde/mod.rs

sed -i 's/pub use crate::dae::{DAE, DAEProblem};/pub use crate::dae::DAE;/g' src/prelude.rs
sed -i 's/pub use crate::dde::{DDE, DDEProblem};/pub use crate::dde::DDE;/g' src/prelude.rs
sed -i 's/pub use crate::ode::{ODE, ODEProblem};/pub use crate::ode::ODE;/g' src/prelude.rs
sed -i 's/pub use crate::sde::{SDE, SDEProblem};/pub use crate::sde::SDE;/g' src/prelude.rs
echo "pub use crate::ivp::Ivp;" >> src/prelude.rs

rm src/ode/problem.rs src/dae/problem.rs src/sde/problem.rs src/dde/problem.rs

# Patch default solout
sed -i 's/pub struct DefaultSolout {}/#[derive(Clone, Debug)]\npub struct DefaultSolout {}/g' src/solout/default.rs

find examples tests -type f -name "*.rs" -exec sed -i -e 's/ODEProblem::new(/Ivp::ode(/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/DAEProblem::new(/Ivp::dae(\&/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/SDEProblem::new(/Ivp::sde(/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/DDEProblem::new(/Ivp::dde(/g' {} +

find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut solver)/.method(solver.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut method)/.method(method.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut var_solver)/.method(var_solver.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut reference_solver)/.method(reference_solver.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut solver_dense)/.method(solver_dense.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut solver_even)/.method(solver_even.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut solver_t_out)/.method(solver_t_out.clone()).solve()/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.solve(&mut ExplicitRungeKutta::rk4(0.01))/.method(ExplicitRungeKutta::rk4(0.01_f64)).solve()/g' {} +

sed -i 's/let solution = problem.method(method.clone()).solve().unwrap();/let solution = problem.clone().method(method.clone()).solve().unwrap();/g' examples/ode/06_integration/main.rs
sed -i 's/let ivp_dense = problem.dense(2);/let ivp_dense = problem.clone().dense(2);/g' examples/ode/06_integration/main.rs
sed -i 's/let ivp_even = problem.even(1.0);/let ivp_even = problem.clone().even(1.0);/g' examples/ode/06_integration/main.rs

find examples tests -type f -name "*.rs" -exec grep -q "Ivp::" {} \; -exec sed -i -e '/use differential_equations::prelude::\*/a use differential_equations::ivp::Ivp;' {} +

sed -i 's/let tf = 5.0;/let tf: f64 = 5.0;/g' examples/ode/06_integration/main.rs

sed -i 's/\/\/\/ Defines systems to test the ODE solvers/\/\/ Defines systems to test the ODE solvers/g' tests/ode/systems.rs
sed -i 's/\/\/! Defines systems to test the ODE solvers/\/\/ Defines systems to test the ODE solvers/g' tests/ode/systems.rs
sed -i 's/\/\/\/ Suite of test cases for checking the interpolation of ODE solvers./\/\/ Suite of test cases for checking the interpolation of ODE solvers./g' tests/ode/interpolation.rs
sed -i 's/\/\/\/ Suite of test cases for checking the accuracy of ODE solvers./\/\/ Suite of test cases for checking the accuracy of ODE solvers./g' tests/ode/accuracy.rs
sed -i 's/\/\/\/ Test for comparing solver accuracy /\/\/ Test for comparing solver accuracy /g' tests/ode/comparison.rs
sed -i 's/\/\/\/ Tests for gracefully handling errors /\/\/ Tests for gracefully handling errors /g' tests/ode/errors.rs
sed -i 's/\/\/! Suite of test cases for ODE NumericalMethods./\/\/ Suite of test cases for ODE NumericalMethods./g' tests/ode/accuracy.rs
sed -i 's/\/\/! Expected results should be verified against a trusted solver./\/\/ Expected results should be verified against a trusted solver./g' tests/ode/accuracy.rs
sed -i 's/\/\/! Suite of test cases for numerical methods vs results of SciPy using DOP853 \& Tolerences = 1e-12/\/\/ Suite of test cases for numerical methods vs results of SciPy using DOP853 \& Tolerences = 1e-12/g' tests/ode/accuracy.rs
sed -i 's/\/\/! Compares the performance of solvers by the statistics, i.e. number of steps, function evaluations, etc./\/\/ Compares the performance of solvers by the statistics, i.e. number of steps, function evaluations, etc./g' tests/ode/comparison.rs
sed -i 's/\/\/! Suite of test cases for numerical method error handling/\/\/ Suite of test cases for numerical method error handling/g' tests/ode/errors.rs
sed -i 's/\/\/! Suite of test cases for checking the interpolation of the solvers./\/\/ Suite of test cases for checking the interpolation of the solvers./g' tests/ode/interpolation.rs

sed -i 's/\/\/\/ Expected results should be verified against a trusted solver./\/\/ Expected results should be verified against a trusted solver./g' tests/dde/accuracy.rs
sed -i 's/\/\/\/ Suite of test cases for checking the interpolation of DDE solvers./\/\/ Suite of test cases for checking the interpolation of DDE solvers./g' tests/dde/interpolation.rs
sed -i 's/\/\/\/ Defines systems to test the DDE solvers/\/\/ Defines systems to test the DDE solvers/g' tests/dde/systems.rs
sed -i 's/\/\/\/ Suite of test cases for DDE NumericalMethods./\/\/ Suite of test cases for DDE NumericalMethods./g' tests/dde/accuracy.rs
sed -i 's/\/\/! Defines systems to test the DDE solvers/\/\/ Defines systems to test the DDE solvers/g' tests/dde/systems.rs
sed -i 's/\/\/! Suite of test cases for checking the interpolation of DDE solvers./\/\/ Suite of test cases for checking the interpolation of DDE solvers./g' tests/dde/interpolation.rs
sed -i 's/\/\/! Suite of test cases for DDE NumericalMethods./\/\/ Suite of test cases for DDE NumericalMethods./g' tests/dde/accuracy.rs
sed -i 's/\/\/! Expected results should be verified against a trusted solver./\/\/ Expected results should be verified against a trusted solver./g' tests/dde/accuracy.rs

find tests/ode tests/dae tests/dde tests/sde -type f -name "*.rs" -exec sed -i -e '1i\use differential_equations::ivp::Ivp;' {} +
sed -i 's/ode::ODEProblem,//g' tests/ode/accuracy.rs
sed -i 's/ode::ODEProblem,//g' tests/ode/interpolation.rs
sed -i 's/ode::{ODE, ODEProblem}/ode::ODE/g' tests/ode/errors.rs
sed -i 's/ode::ODEProblem//g' tests/ode/comparison.rs

find examples tests -type f -name "*.rs" -exec sed -i -e 's/let sol = problem.clone().method(solver.clone()).solve().unwrap();/let sol = problem.clone().method(solver).solve().unwrap();/g' {} +
sed -i 's/\.solout(\&mut solout)/.solout(solout)/g' examples/ode/10_custom_solout/main.rs
sed -i 's/use differential_equations::ivp::Ivp;;//g' examples/ode/10_custom_solout/main.rs
sed -i 's/use differential_equations::{dde::DDEProblem/use differential_equations::{dde::DDE/g' tests/dde/accuracy.rs
sed -i 's/use differential_equations::{dde::DDEProblem/use differential_equations::{dde::DDE/g' tests/dde/interpolation.rs

sed -i 's/let sol = problem.method(solver.clone()).solve().unwrap();/let sol = problem.clone().method(solver.clone()).solve().unwrap();/g' tests/ode/comparison.rs

sed -i 's/(yf\[i\] - $expected_result\[i\]).abs() < $tolerance/(yf\[i\] - $expected_result\[i\]).abs() < $tolerance as f64/g' tests/ode/accuracy.rs
sed -i 's/(yf\[i\] - $expected_result\[i\]).abs() < $tolerance as f64/(yf\[i\] - $expected_result\[i\] as f64).abs() < $tolerance as f64/g' tests/ode/accuracy.rs

sed -i '/use differential_equations::ivp::Ivp;/d' tests/dde/main.rs
sed -i '/use differential_equations::ivp::Ivp;/d' tests/dde/systems.rs

sed -i 's/let mut method =/let method =/g' examples/ode/04_sir_model/main.rs
sed -i 's/let mut method =/let method =/g' examples/ode/10_custom_solout/main.rs
sed -i 's/let mut method =/let method =/g' examples/dae/01_amplifier/main.rs
sed -i 's/let mut solver =/let solver =/g' tests/dde/accuracy.rs
sed -i 's/let mut solver =/let solver =/g' tests/dde/interpolation.rs

sed -i 's/let sol_flt = problem.clone().method(ExplicitRungeKutta::dop853().rtol(1e-12).atol(1e-12)).solve().unwrap();/let sol_flt = problem.clone().method(var_solver).solve().unwrap();/g' examples/ode/14_r2bp_stm/main.rs
sed -i 's/let mut var_solver = ExplicitRungeKutta::dop853().atol(1e-14).rtol(1e-14);/let mut var_solver = ExplicitRungeKutta::dop853().atol(1e-14).rtol(1e-14);/g' examples/ode/14_r2bp_stm/main.rs

sed -i 's/let result = problem\n        \/\/ The custom solout is applied to the problem here\n        \.solout(solout)\n        \.method(solver)\.solve()/let result = problem.clone()\n        \/\/ The custom solout is applied to the problem here\n        \.solout(solout.clone())\n        \.method(solver)\.solve()/g' examples/ode/10_custom_solout/main.rs
sed -i 's/struct PendulumSolout/#[derive(Clone)]\nstruct PendulumSolout/g' examples/ode/10_custom_solout/main.rs
sed -i 's/match Ivp::ode(&HarmonicOscillator { k: 1.0 }, 0.0, 10.0, vector!\[1.0, 0.0\])\n            .solve(&mut ExplicitRungeKutta::rk4(0.01))/match Ivp::ode(\&HarmonicOscillator { k: 1.0 }, 0.0, 10.0, vector!\[1.0, 0.0\])\n            .method(ExplicitRungeKutta::rk4(0.01_f64)).solve()/g' examples/ode/02_harmonic_oscillator/main.rs
sed -i 's/let solution =/let solution: Solution<f64, _> =/g' examples/ode/02_harmonic_oscillator/main.rs
sed -i 's/let mut method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);/let method = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);/g' examples/ode/09_matrix_ode/main.rs
sed -i 's/for (i, (t, y)) in solution.iter().enumerate()/for (i, (\&t, y)) in solution.iter().enumerate()/g' examples/ode/09_matrix_ode/main.rs
sed -i 's/let sol_flt = problem.clone().method(var_solver).solve().unwrap();/let sol_flt = problem.clone().method(var_solver.clone()).solve().unwrap();/g' examples/ode/14_r2bp_stm/main.rs
