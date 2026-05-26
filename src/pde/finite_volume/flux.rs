//! Numerical fluxes for finite volume methods.

use crate::{
    pde::PDE,
    traits::{Real, State},
};

/// Numerical flux function for Riemann problems.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NumericalFlux {
    /// Central flux (unstable for pure advection, useful as a baseline).
    #[default]
    Central,
    /// Upwind flux (requires characteristic information, currently generic advection placeholder).
    Upwind,
    /// Harten-Lax-van Leer-Contact (HLLC) approximate Riemann solver.
    Hllc,
    /// Local Lax-Friedrichs (Rusanov) numerical flux.
    Rusanov,
}

impl NumericalFlux {
    /// Compute the numerical flux at an interface.
    ///
    /// For many PDE systems, you only have a generic flux function `f(u)`. Central flux is generic.
    /// Rusanov (Local Lax-Friedrichs) requires a maximum wave speed `s_max` which we approximate
    /// by a user-provided global/local value, but without a full eigen-decomposition it's generic.
    /// HLLC and Upwind usually require system-specific Riemann solvers. We will implement basic
    /// generic forms where possible, but allow users to implement specific Riemann solvers inside
    /// their `PDE` implementation if needed. For now, we provide Central and a basic Rusanov.
    pub fn compute<Eq, T, U, const D: usize>(
        &self,
        equation: &Eq,
        t: T,
        x: &[T; D],
        u_l: &U,
        u_r: &U,
        axis: usize,
    ) -> U
    where
        T: Real,
        U: State<T>,
        Eq: PDE<T, U, D> + ?Sized,
    {
        match self {
            Self::Central => {
                let mut flux_l = core::array::from_fn(|_| u_l.zeros_like());
                let mut flux_r = core::array::from_fn(|_| u_r.zeros_like());
                let grad_zero = core::array::from_fn(|_| u_l.zeros_like());

                equation.flux(t, x, u_l, &grad_zero, &mut flux_l);
                equation.flux(t, x, u_r, &grad_zero, &mut flux_r);

                let mut central_flux = u_l.zeros_like();
                let half = T::from_subset(&0.5);
                for i in 0..u_l.len() {
                    central_flux.set_component(
                        i,
                        half * (flux_l[axis].get_component(i) + flux_r[axis].get_component(i)),
                    );
                }
                central_flux
            }
            Self::Upwind => {
                // Placeholder: Upwind generally requires knowing the wave direction.
                // We'll fall back to Central for generic PDEs without wave speed info.
                Self::Central.compute(equation, t, x, u_l, u_r, axis)
            }
            Self::Hllc => {
                // Placeholder: HLLC requires wave speeds (S_L, S_M, S_R) and specific
                // system knowledge (like Euler equations).
                // We'll fall back to Central for generic PDEs.
                Self::Central.compute(equation, t, x, u_l, u_r, axis)
            }
            Self::Rusanov => {
                // Rusanov flux: F_{i+1/2} = 0.5 * (F_L + F_R) - 0.5 * S_max * (U_R - U_L)
                // For a truly generic Rusanov, we need max wave speed `s_max`.
                // As a fallback, we compute the flux jacobian spectral radius numerically,
                // or just use a fixed max speed if not provided. We'll use a simplified
                // approach here or just fall back to central if wave speed isn't available.
                // We will add a placeholder generic implementation.
                let mut flux_l = core::array::from_fn(|_| u_l.zeros_like());
                let mut flux_r = core::array::from_fn(|_| u_r.zeros_like());
                let grad_zero = core::array::from_fn(|_| u_l.zeros_like());

                equation.flux(t, x, u_l, &grad_zero, &mut flux_l);
                equation.flux(t, x, u_r, &grad_zero, &mut flux_r);

                // TODO: Need a way to compute or get s_max. We'll use a placeholder value of 1.0
                // for generic systems if they don't provide it.
                let s_max = T::one(); // Needs characteristic speed from PDE trait.

                let mut rusanov_flux = u_l.zeros_like();
                let half = T::from_subset(&0.5);
                for i in 0..u_l.len() {
                    let f_l = flux_l[axis].get_component(i);
                    let f_r = flux_r[axis].get_component(i);
                    let du = u_r.get_component(i) - u_l.get_component(i);
                    rusanov_flux.set_component(i, half * (f_l + f_r - s_max * du));
                }
                rusanov_flux
            }
        }
    }
}
