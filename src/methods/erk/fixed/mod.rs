//! Runge-Kutta solvers with fixed step-size.

mod ordinary;
mod delay;

use super::ExplicitRungeKutta;
use crate::methods::{Fixed, Ordinary, Delay};

use crate::{
    traits::{CallBackData, Real, State},
    tableau::ButcherTableau,
};

// Macro for fixed step constructors
macro_rules! impl_erk_fixed_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<E, Fixed, T, V, D, $s_val, $s_val> {
            #[doc = $doc]
            pub fn $method_name(h0: T) -> Self {
                let order = $order_val;
                let tableau = ButcherTableau::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;

                ExplicitRungeKutta {
                    h0,
                    c,
                    a,
                    b,
                    order,
                    ..Default::default()
                }
            }
        }
    };
}

// Fixed step methods (S = I, no embedded error estimation, cubic Hermite interpolation)
impl_erk_fixed_step_constructor!(euler, 1, 1, "Creates an Explicit Euler method (1st order, 1 stage).");
impl_erk_fixed_step_constructor!(midpoint, 2, 2, "Creates an Explicit Midpoint method (2nd order, 2 stages).");
impl_erk_fixed_step_constructor!(heun, 2, 2, "Creates an Explicit Heun method (2nd order, 2 stages).");
impl_erk_fixed_step_constructor!(ralston, 2, 2, "Creates an Explicit Ralston method (2nd order, 2 stages).");
impl_erk_fixed_step_constructor!(rk4, 4, 4, "Creates the classical 4th order Runge-Kutta method.");
impl_erk_fixed_step_constructor!(three_eighths, 4, 4, "Creates the three-eighths rule 4th order Runge-Kutta method.");