//! Runge-Kutta solvers with support for dense output, embedded error estimation, and fixed steps.

mod delay;
mod ordinary;

use crate::{
    methods::{Adaptive, ExplicitRungeKutta},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Macro for adaptive step constructors
macro_rules! impl_erk_adaptive_step_constructor {
    ($method_name:ident, $fsal_val:expr, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData>
            ExplicitRungeKutta<E, Adaptive, T, V, D, $order_val, $s_val, $m_val>
        {
            #[doc = $doc]
            pub fn $method_name() -> Self {
                let tableau = ButcherTableau::<T, $s_val, $m_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let bh = tableau.bh;
                let bi = tableau.bi;
                let fsal = $fsal_val;

                ExplicitRungeKutta {
                    c,
                    a,
                    b,
                    bh,
                    bi,
                    fsal,
                    ..Default::default()
                }
            }
        }
    };
}

// Adaptive step methods (embedded error estimation, cubic Hermite interpolation)
impl_erk_adaptive_step_constructor!(
    rkf45,
    false,
    5,
    6,
    6,
    "Creates a Runge-Kutta-Fehlberg 4(5) method with error estimation."
);
impl_erk_adaptive_step_constructor!(
    cash_karp,
    false,
    5,
    6,
    6,
    "Creates a Cash-Karp 4(5) method with error estimation."
);
impl_erk_adaptive_step_constructor!(
    rkv655e,
    true,
    6,
    9,
    10,
    "Creates a ExplictRungeKutta 6(5) method with 9 stages and a 5th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv656e,
    true,
    6,
    9,
    12,
    "Creates a ExplictRungeKutta 6(5) method with 9 stages and a 6th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv766e,
    false,
    7,
    10,
    13,
    "Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 6th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv767e,
    false,
    7,
    10,
    16,
    "Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 7th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv877e,
    false,
    8,
    13,
    17,
    "Creates a ExplicitRungeKutta 8(7) method with 13 stages and a 7th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv878e,
    false,
    8,
    13,
    21,
    "Creates a ExplicitRungeKutta 8(7) method with 13 stages and a 8th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv988e,
    false,
    9,
    16,
    21,
    "Creates a ExplicitRungeKutta 9(8) method with 16 stages and a 8th order interpolant."
);
impl_erk_adaptive_step_constructor!(
    rkv989e,
    false,
    9,
    16,
    26,
    "Creates a ExplicitRungeKutta 9(8) method with 16 stages and a 9th order interpolant."
);
