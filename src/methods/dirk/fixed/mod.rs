//! Diagonally Implicit Runge-Kutta solvers with Newton iteration for each stage.

mod ordinary;

use crate::{
    methods::{DiagonallyImplicitRungeKutta, Fixed},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Macro for fixed step constructors
macro_rules! impl_dirk_fixed_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, Y: State<T>, D: CallBackData>
            DiagonallyImplicitRungeKutta<E, Fixed, T, Y, D, $order_val, $s_val, $m_val>
        {
            #[doc = $doc]
            pub fn $method_name(h0: T) -> Self {
                let tableau = ButcherTableau::<T, $s_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;

                DiagonallyImplicitRungeKutta {
                    h0,
                    c,
                    a,
                    b,
                    order: $order_val,
                    stages: $s_val,
                    ..Default::default()
                }
            }
        }
    };
}

// Fixed step DIRK methods
impl_dirk_fixed_step_constructor!(
    esdirk33,
    3,
    3,
    3,
    "ESDIRK-3-3: 3-stage, 3rd order ESDIRK method."
);
