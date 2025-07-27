//! Implicit Runge-Kutta solvers with Newton iteration and adaptive step size control.

mod ordinary;
//mod delay;

use crate::{
    methods::{ImplicitRungeKutta, Adaptive},
    traits::{CallBackData, Real, State},
    tableau::ButcherTableau,
};

// Macro for adaptive step constructors
macro_rules! impl_irk_adaptive_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData> ImplicitRungeKutta<E, Adaptive, T, V, D, $order_val, $s_val, $m_val> {
            #[doc = $doc]
            pub fn $method_name() -> Self {
                let tableau = ButcherTableau::<T, $s_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let bh = tableau.bh;
                
                ImplicitRungeKutta {
                    c,
                    a,
                    b,
                    bh,
                    order: $order_val,
                    stages: $s_val,
                    dense_stages: $m_val,
                    ..Default::default()
                }
            }
        }
    };
}

// Adaptive step methods (embedded error estimation, Newton iteration)

// Gauss-Legendre methods - A-stable, symplectic, highly accurate
impl_irk_adaptive_step_constructor!(gauss_legendre_4, 4, 2, 2, "Creates a new Gauss-Legendre 2-stage implicit Runge-Kutta method of order 4.");
impl_irk_adaptive_step_constructor!(gauss_legendre_6, 6, 3, 3, "Creates a new Gauss-Legendre 3-stage implicit Runge-Kutta method of order 6.");

// Lobatto IIIC methods - L-stable, algebraically stable  
impl_irk_adaptive_step_constructor!(lobatto_iiic_2, 2, 2, 2, "Creates a new Lobatto IIIC 2-stage implicit Runge-Kutta method of order 2.");
impl_irk_adaptive_step_constructor!(lobatto_iiic_4, 4, 3, 3, "Creates a new Lobatto IIIC 3-stage implicit Runge-Kutta method of order 4.");