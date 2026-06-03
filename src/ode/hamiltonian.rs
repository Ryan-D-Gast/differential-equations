use crate::{
    ode::ODE,
    traits::{DefaultState, Real, State},
};

/// Hamiltonian system trait for symplectic solvers.
///
/// Hamiltonian systems are defined by two coupled first-order differential equations
/// governing coordinates $q$ and momenta $p$:
///
/// dq/dt = velocity(t, q, p)
/// dp/dt = force(t, q, p)
///
/// The state vector for the ODE solver is assumed to be laid out as `y = [q, p]`.
pub trait Hamiltonian<T = f64, Y = DefaultState<T>>
where
    T: Real,
    Y: State<T>,
{
    /// Compute the time derivative of positions: dq/dt = velocity(t, q, p)
    fn velocity(&self, t: T, q: &Y, p: &Y, dq: &mut Y);

    /// Compute the time derivative of momenta: dp/dt = force(t, q, p)
    fn force(&self, t: T, q: &Y, p: &Y, dp: &mut Y);
}

impl<H, T, Y> Hamiltonian<T, Y> for &H
where
    T: Real,
    Y: State<T>,
    H: Hamiltonian<T, Y> + ?Sized,
{
    fn velocity(&self, t: T, q: &Y, p: &Y, dq: &mut Y) {
        (*self).velocity(t, q, p, dq);
    }

    fn force(&self, t: T, q: &Y, p: &Y, dp: &mut Y) {
        (*self).force(t, q, p, dp);
    }
}

/// Adapter that wraps a [`Hamiltonian`] and implements [`ODE`] for generic states.
#[derive(Clone, Debug)]
pub struct HamiltonianSystem<H> {
    hamiltonian: H,
}

impl<H> HamiltonianSystem<H> {
    /// Create a new ODE adapter for the given Hamiltonian.
    pub fn new(hamiltonian: H) -> Self {
        Self { hamiltonian }
    }
}

impl<H, T, Y> ODE<T, Y> for HamiltonianSystem<H>
where
    T: Real,
    Y: State<T>,
    H: Hamiltonian<T, Y>,
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        let n = y.len();
        let half = n / 2;

        let mut q = y.zeros_like();
        let mut p = y.zeros_like();
        for i in 0..half {
            q.set_component(i, y.get_component(i));
            p.set_component(i, y.get_component(half + i));
        }

        let mut dq = y.zeros_like();
        let mut dp = y.zeros_like();

        self.hamiltonian.velocity(t, &q, &p, &mut dq);
        self.hamiltonian.force(t, &q, &p, &mut dp);

        for i in 0..half {
            dydt.set_component(i, dq.get_component(i));
            dydt.set_component(half + i, dp.get_component(i));
        }
    }
}

/// Wrapper to construct a Hamiltonian system from closures.
#[derive(Clone, Debug)]
pub struct HamiltonianFnWrapper<V, F> {
    velocity: V,
    force: F,
}

impl<V, F> HamiltonianFnWrapper<V, F> {
    /// Create a new Hamiltonian wrapper from a velocity closure and a force closure.
    pub fn new(velocity: V, force: F) -> Self {
        Self { velocity, force }
    }
}

impl<T, Y, V, F> Hamiltonian<T, Y> for HamiltonianFnWrapper<V, F>
where
    T: Real,
    Y: State<T>,
    V: Fn(T, &Y, &Y, &mut Y),
    F: Fn(T, &Y, &Y, &mut Y),
{
    fn velocity(&self, t: T, q: &Y, p: &Y, dq: &mut Y) {
        (self.velocity)(t, q, p, dq);
    }

    fn force(&self, t: T, q: &Y, p: &Y, dp: &mut Y) {
        (self.force)(t, q, p, dp);
    }
}
