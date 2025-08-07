//! Runge-Kutta solvers with fixed step-size.

mod delay;
mod ordinary;
mod stochastic;

use crate::{
    methods::{ExplicitRungeKutta, Fixed},
    tableau::ButcherTableau,
    traits::{CallBackData, Real, State},
};

// Macro for fixed step constructors
macro_rules! impl_erk_fixed_step_constructor {
    ($method_name:ident, $fsal_val:expr, $order_val:expr, $s_val:expr, $doc:expr) => {
        impl<E, T: Real, Y: State<T>, D: CallBackData>
            ExplicitRungeKutta<E, Fixed, T, Y, D, $order_val, $s_val, $s_val>
        {
            #[doc = $doc]
            pub fn $method_name(h0: T) -> Self {
                let tableau = ButcherTableau::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let fsal = $fsal_val;

                ExplicitRungeKutta {
                    h0,
                    c,
                    a,
                    b,
                    fsal,
                    ..Default::default()
                }
            }
        }
    };
}

// Fixed step methods (S = I, no embedded error estimation, cubic Hermite interpolation)
impl_erk_fixed_step_constructor!(
    euler,
    false,
    1,
    1,
    "Creates an Explicit Euler method (1st order, 1 stage)."
);
impl_erk_fixed_step_constructor!(
    midpoint,
    false,
    2,
    2,
    "Creates an Explicit Midpoint method (2nd order, 2 stages)."
);
impl_erk_fixed_step_constructor!(
    heun,
    false,
    2,
    2,
    "Creates an Explicit Heun method (2nd order, 2 stages)."
);
impl_erk_fixed_step_constructor!(
    ralston,
    false,
    2,
    2,
    "Creates an Explicit Ralston method (2nd order, 2 stages)."
);
impl_erk_fixed_step_constructor!(
    rk4,
    false,
    4,
    4,
    "Creates the classical 4th order Runge-Kutta method."
);
impl_erk_fixed_step_constructor!(
    three_eighths,
    false,
    4,
    4,
    "Creates the three-eighths rule 4th order Runge-Kutta method."
);
