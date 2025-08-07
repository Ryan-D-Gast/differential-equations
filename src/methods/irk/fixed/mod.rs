//! Implicit Runge-Kutta solvers with Newton iteration.

mod ordinary;
//mod delay;

use crate::{
    methods::{Fixed, ImplicitRungeKutta},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Macro for fixed step constructors
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

// Fixed step methods (embedded error estimation, Newton iteration)
impl_irk_fixed_step_constructor!(
    backward_euler,
    1,
    1,
    1,
    "Backward Euler method of order 1 with 1 stage. A-stable and suitable for stiff problems."
);
impl_irk_fixed_step_constructor!(
    crank_nicolson,
    2,
    2,
    2,
    "Crank-Nicolson method of order 2 with 2 stages. A-stable and suitable for stiff problems."
);
impl_irk_fixed_step_constructor!(
    trapezoidal,
    2,
    2,
    2,
    "Trapezoidal method of order 2 with 2 stages. A-stable and suitable for stiff problems."
);
impl_irk_fixed_step_constructor!(
    radau_iia_3,
    3,
    2,
    2,
    "Radau IIA method of order 3 with 2 stages. A-stable and suitable for stiff problems."
);
impl_irk_fixed_step_constructor!(
    radau_iia_5,
    5,
    3,
    3,
    "Radau IIA method of order 5 with 3 stages. A-stable and suitable for stiff problems."
);
