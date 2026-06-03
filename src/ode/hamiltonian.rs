use crate::{
    ode::ODE,
    traits::{Real, State},
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
pub trait Hamiltonian<T = f64>
where
    T: Real,
{
    /// Compute the time derivative of positions: dq/dt = velocity(t, q, p)
    fn velocity(&self, t: T, q: &[T], p: &[T], dq: &mut [T]);

    /// Compute the time derivative of momenta: dp/dt = force(t, q, p)
    fn force(&self, t: T, q: &[T], p: &[T], dp: &mut [T]);
}

impl<H, T> Hamiltonian<T> for &H
where
    T: Real,
    H: Hamiltonian<T> + ?Sized,
{
    fn velocity(&self, t: T, q: &[T], p: &[T], dq: &mut [T]) {
        (*self).velocity(t, q, p, dq);
    }

    fn force(&self, t: T, q: &[T], p: &[T], dp: &mut [T]) {
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
    H: Hamiltonian<T>,
{
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        let n = y.len();
        let half = n / 2;

        let mut q = vec![T::zero(); half];
        let mut p = vec![T::zero(); half];
        for i in 0..half {
            q[i] = y.get_component(i);
            p[i] = y.get_component(half + i);
        }

        let mut dq = vec![T::zero(); half];
        let mut dp = vec![T::zero(); half];

        self.hamiltonian.velocity(t, &q, &p, &mut dq);
        self.hamiltonian.force(t, &q, &p, &mut dp);

        for i in 0..half {
            dydt.set_component(i, dq[i]);
            dydt.set_component(half + i, dp[i]);
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

impl<T, V, F> Hamiltonian<T> for HamiltonianFnWrapper<V, F>
where
    T: Real,
    V: Fn(T, &[T], &[T], &mut [T]),
    F: Fn(T, &[T], &[T], &mut [T]),
{
    fn velocity(&self, t: T, q: &[T], p: &[T], dq: &mut [T]) {
        (self.velocity)(t, q, p, dq);
    }

    fn force(&self, t: T, q: &[T], p: &[T], dp: &mut [T]) {
        (self.force)(t, q, p, dp);
    }
}
