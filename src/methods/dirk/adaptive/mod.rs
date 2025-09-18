//! DIRK with Newton solves and adaptive step size.

mod ordinary;

use crate::{
    methods::{Adaptive, DiagonallyImplicitRungeKutta},
    tableau::ButcherTableau,
    traits::{Real, State},
};

// Adaptive step constructors
macro_rules! impl_dirk_adaptive_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, Y: State<T>>
            DiagonallyImplicitRungeKutta<E, Adaptive, T, Y, $order_val, $s_val, $m_val>
        {
            #[doc = $doc]
            pub fn $method_name() -> Self {
                let tableau = ButcherTableau::<T, $s_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let bh = tableau.bh;

                DiagonallyImplicitRungeKutta {
                    c,
                    a,
                    b,
                    bh,
                    order: $order_val,
                    stages: $s_val,
                    ..Default::default()
                }
            }
        }
    };
}

// DIRK methods with embedded error estimation
impl_dirk_adaptive_step_constructor!(
    sdirk21,
    2,
    2,
    2,
    "SDIRK-2-1: 2-stage, order 2(1). Embedded; L-stable."
);
impl_dirk_adaptive_step_constructor!(
    esdirk324l2sa,
    3,
    4,
    4,
    "ESDIRK3(2)4L[2]SA: 4-stage, order 3(2). Embedded; L-stable."
);
impl_dirk_adaptive_step_constructor!(
    kvaerno423,
    3,
    4,
    4,
    "Kvaerno(4,2,3): 4-stage, order 3(2). Embedded; L-stable."
);
impl_dirk_adaptive_step_constructor!(
    kvaerno745,
    5,
    7,
    7,
    "Kvaerno(7,4,5): 7-stage, order 5(4). Embedded; L-stable."
);
