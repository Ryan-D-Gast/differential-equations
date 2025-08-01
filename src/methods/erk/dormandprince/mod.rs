//! Runge-Kutta solvers with support for dense output, embedded error estimation, and fixed steps.

mod delay;
mod ordinary;

use crate::{
    methods::{DormandPrince, ExplicitRungeKutta},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Macro for adaptive step constructors
macro_rules! impl_erk_dormand_prince_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData>
            ExplicitRungeKutta<E, DormandPrince, T, V, D, $order_val, $s_val, $m_val>
        {
            #[doc = $doc]
            pub fn $method_name() -> Self {
                let tableau = ButcherTableau::<T, $s_val, $m_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let bh = tableau.bh;
                let er = tableau.er;
                let bi = tableau.bi;
                let fsal = true; // DOP methods are FSAL by definition, also this isn't used in implementation because its assumed but just for consistency

                ExplicitRungeKutta {
                    c,
                    a,
                    b,
                    bh,
                    er,
                    bi,
                    fsal,
                    ..Default::default()
                }
            }
        }
    };
}

// Adaptive step methods (embedded error estimation, cubic Hermite interpolation)
impl_erk_dormand_prince_constructor!(
    dop853,
    8,
    12,
    16,
    "Creates the DOP853 method (8th order, 12 stages, 4 dense output stages)."
);
impl_erk_dormand_prince_constructor!(
    dopri5,
    5,
    7,
    7,
    "Creates the DOPRI5 method (5th order, 7 stages)."
);
