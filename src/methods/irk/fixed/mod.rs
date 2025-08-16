//! IRK with Newton solves (fixed step).

mod ordinary;

use crate::{
    methods::{Fixed, ImplicitRungeKutta},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Fixed-step constructors
macro_rules! impl_irk_fixed_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, Y: State<T>, D: CallBackData>
            ImplicitRungeKutta<E, Fixed, T, Y, D, $order_val, $s_val, $m_val>
        {
            #[doc = $doc]
            pub fn $method_name(h0: T) -> Self {
                let tableau = ButcherTableau::<T, $s_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;

                ImplicitRungeKutta {
                    h0,
                    c,
                    a,
                    b,
                    order: $order_val,
                    stages: $s_val,
                    dense_stages: $m_val,
                    ..Default::default()
                }
            }
        }
    };
}

// Fixed-step IRK methods
impl_irk_fixed_step_constructor!(
    backward_euler,
    1,
    1,
    1,
    "Backward Euler, order 1, 1 stage. A-stable; stiff-suitable."
);
impl_irk_fixed_step_constructor!(
    crank_nicolson,
    2,
    2,
    2,
    "Crank-Nicolson, order 2, 2 stages. A-stable; stiff-suitable."
);
impl_irk_fixed_step_constructor!(
    trapezoidal,
    2,
    2,
    2,
    "Trapezoidal, order 2, 2 stages. A-stable; stiff-suitable."
);