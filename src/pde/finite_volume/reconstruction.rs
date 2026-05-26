//! Reconstruction methods for finite volume methods.

use super::Limiter;
use crate::traits::{Real, State};

/// Spatial reconstruction options for cell-centered finite volumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reconstruction {
    /// Piecewise constant (first order).
    #[default]
    Constant,
    /// Piecewise linear with MUSCL extrapolation (second order).
    MuscL,
}

impl Reconstruction {
    /// Reconstruct the left and right states at a cell interface.
    ///
    /// Given the cell averages `u_ll`, `u_l`, `u_r`, `u_rr` (where the interface is between `l` and `r`),
    /// this computes the left state `u_face_l` and the right state `u_face_r` at the interface.
    /// `dx` is the cell size.
    pub fn reconstruct<T, U>(
        &self,
        u_ll: &U,
        u_l: &U,
        u_r: &U,
        u_rr: &U,
        limiter: &Limiter,
    ) -> (U, U)
    where
        T: Real,
        U: State<T>,
    {
        let mut u_face_l = u_l.clone();
        let mut u_face_r = u_r.clone();

        match self {
            Self::Constant => {
                // Piecewise constant: face states are just the cell averages.
                (u_face_l, u_face_r)
            }
            Self::MuscL => {
                let zero = T::zero();
                let half = T::from_subset(&0.5);
                let epsilon = T::from_subset(&1e-12); // To avoid division by zero

                for i in 0..u_l.len() {
                    let dll = u_l.get_component(i) - u_ll.get_component(i);
                    let dl = u_r.get_component(i) - u_l.get_component(i);
                    let dr = u_rr.get_component(i) - u_r.get_component(i);

                    // Reconstruct left state at interface (i+1/2)
                    let r_l = if dl.abs() > epsilon { dll / dl } else { zero };
                    let phi_l = limiter.compute(r_l);
                    u_face_l.set_component(i, u_l.get_component(i) + half * phi_l * dl);

                    // Reconstruct right state at interface (i+1/2)
                    let r_r = if dl.abs() > epsilon { dr / dl } else { zero };
                    let phi_r = limiter.compute(r_r);
                    u_face_r.set_component(i, u_r.get_component(i) - half * phi_r * dl);
                }
                (u_face_l, u_face_r)
            }
        }
    }
}
