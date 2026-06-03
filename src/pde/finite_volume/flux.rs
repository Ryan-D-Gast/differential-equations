//! Numerical fluxes for finite volume methods.

use crate::{
    pde::PDE,
    traits::{Real, State},
};

/// Numerical flux function for finite-volume interfaces.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum NumericalFlux {
    /// Central flux (unstable for pure advection, useful as a baseline).
    #[default]
    Central,
    /// Local Lax-Friedrichs (Rusanov) numerical flux with a maximum wave speed.
    Rusanov {
        /// Upper bound on the characteristic speed at the interface.
        max_speed: f64,
    },
}

impl NumericalFlux {
    /// Compute the numerical flux at an interface.
    ///
    /// Central flux is generic. Rusanov requires the caller to provide a maximum wave speed;
    /// this keeps the generic PDE trait honest because it does not expose characteristic speeds.
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
            Self::Rusanov { max_speed } => {
                // Rusanov flux: F_{i+1/2} = 0.5 * (F_L + F_R) - 0.5 * S_max * (U_R - U_L)
                let mut flux_l = core::array::from_fn(|_| u_l.zeros_like());
                let mut flux_r = core::array::from_fn(|_| u_r.zeros_like());
                let grad_zero = core::array::from_fn(|_| u_l.zeros_like());

                equation.flux(t, x, u_l, &grad_zero, &mut flux_l);
                equation.flux(t, x, u_r, &grad_zero, &mut flux_r);

                let s_max = T::from_subset(max_speed);

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
