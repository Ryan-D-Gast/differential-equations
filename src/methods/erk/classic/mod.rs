//! Runge-Kutta solvers with support for dense output, embedded error estimation, and fixed steps.

mod ode;
mod dde;

use super::ExplicitRungeKutta;

use crate::{
    traits::{CallBackData, Real, State},
    tableau::ButcherTableau,
};

/// Typestate pattern for Classic Runge-Kutta methods
pub struct Classic;

// Macro for fixed step constructors
macro_rules! impl_erk_fixed_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<E, Classic, T, V, D, $s_val, $s_val> {
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

// Macro for adaptive step constructors
macro_rules! impl_erk_adaptive_step_constructor {
    ($method_name:ident, $order_val:expr, $s_val:expr, $m_val:expr, $doc:expr) => {
        impl<E, T: Real, V: State<T>, D: CallBackData> ExplicitRungeKutta<E, Classic, T, V, D, $s_val, $m_val> {
            #[doc = $doc]
            pub fn $method_name() -> Self {
                let order = $order_val;
                let tableau = ButcherTableau::<T, $s_val, $m_val>::$method_name();
                let c = tableau.c;
                let a = tableau.a;
                let b = tableau.b;
                let bh = tableau.bh;
                let bi = tableau.bi;

                ExplicitRungeKutta {
                    c,
                    a,
                    b,
                    bh,
                    bi,
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

// Adaptive step methods (embedded error estimation, cubic Hermite interpolation)
impl_erk_adaptive_step_constructor!(rkf45, 5, 6, 6, "Creates a Runge-Kutta-Fehlberg 4(5) method with error estimation.");
impl_erk_adaptive_step_constructor!(cash_karp, 5, 6, 6, "Creates a Cash-Karp 4(5) method with error estimation.");
impl_erk_adaptive_step_constructor!(rkv655e, 6, 9, 10, "Creates a Verner's 6(5) method with dense output of order 5.");
impl_erk_adaptive_step_constructor!(rkv656e, 6, 9, 12, "Creates a Verner's 6(5) method with dense output of order 6.");
impl_erk_adaptive_step_constructor!(rkv766e, 7, 10, 13, "Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 6th order interpolant.");
impl_erk_adaptive_step_constructor!(rkv767e, 7, 10, 16, "Creates a ExplicitRungeKutta 7(6) method with 10 stages and a 7th order interpolant.");
impl_erk_adaptive_step_constructor!(rkv877e, 8, 13, 17, "Creates a ExplicitRungeKutta 8(7) method with 13 stages with 7th order interpolant.");
impl_erk_adaptive_step_constructor!(rkv878e, 8, 13, 21, "Creates a ExplicitRungeKutta 8(7) method with 13 stages with 8th order interpolant.");
impl_erk_adaptive_step_constructor!(rkv988e, 9, 16, 21, "Creates a ExplicitRungeKutta 9(8) method with 16 stages with 8th order interpolant.");
impl_erk_adaptive_step_constructor!(rkv989e, 9, 16, 26, "Creates a ExplicitRungeKutta 9(8) method with 16 stages with 9th order interpolant.");