//! Example 14: Restricted Two-Body Problem with State Transition Matrix
//!
//! This example integrates the Kepler two-body problem and propagates the
//! 6x6 state transition matrix (STM) alongside the state. It demonstrates two
//! equivalent ways to obtain sensitivity information:
//!
//! - automatic differentiation with dual numbers
//! - variational equations for the augmented state
//!
//! It also shows the solver `filter` hook, which can transform the accepted
//! step size before it is used internally. That is useful when you want to:
//! - strip derivatives from a dual-number step size so they do not contaminate
//!   the solver state
//! - clamp, quantize, or otherwise normalize step sizes for reproducibility
//!   or custom time-stepping rules

use differential_equations::prelude::*;
use differential_equations::{ode::ODE, traits::Real};
use nalgebra::{Dim, Matrix3, Matrix6, SVector, Vector6, stack};
use num_dual::{Derivative, DualSVec64, DualStruct};
use std::{f64::consts::PI, iter::zip};

/// Keplerian two-body dynamics.
struct TwoBodyODE<T: Real> {
    /// Gravitational parameter.
    mu: T,
}

impl<T> ODE<T, Vector6<T>> for TwoBodyODE<T>
where
    T: Real,
{
    fn diff(&self, _: T, y: &Vector6<T>, dydt: &mut Vector6<T>) {
        let r = y.fixed_rows::<3>(0);
        dydt.fixed_rows_mut::<3>(0).copy_from(&y.fixed_rows::<3>(3));
        dydt.fixed_rows_mut::<3>(3)
            .copy_from(&(r * -self.mu / r.norm().powi(3)));
    }
}

/// Variational form of the two-body problem.
struct VariationalTwoBodyODE {
    /// Gravitational parameter.
    mu: f64,
}

impl ODE<f64, SVector<f64, 42>> for VariationalTwoBodyODE {
    fn diff(&self, _: f64, y: &SVector<f64, 42>, dydt: &mut SVector<f64, 42>) {
        let r = y.fixed_rows::<3>(0);
        let c = self.mu / r.norm().powi(3);

        // State derivatives.
        dydt.fixed_rows_mut::<3>(0).copy_from(&y.fixed_rows::<3>(3));
        dydt.fixed_rows_mut::<3>(3).copy_from(&(r * (-c)));

        // STM derivatives.
        let mut a = Matrix6::<f64>::zeros();
        for i in 0..3 {
            a[(i, i + 3)] = 1.0;
        }
        let g = (r * r.transpose() * (3.0 / r.norm_squared()) - Matrix3::identity()) * c;
        a.fixed_view_mut::<3, 3>(3, 0).copy_from(&g);

        let phi = Matrix6::<f64>::from_iterator(y.fixed_rows::<36>(6).iter().copied());
        let dphi_dt = a * phi;
        dydt.fixed_rows_mut::<36>(6)
            .copy_from_slice(dphi_dt.as_slice());
    }
}

fn get_derivative<const N: usize>(dual: &DualSVec64<N>) -> [f64; N] {
    dual.eps
        .unwrap_generic(Dim::from_usize(N), Dim::from_usize(1))
        .into()
}

fn main() {
    // Initial state: periapsis of an elliptical orbit with eccentricity 0.5.
    let y0 = Vector6::<f64>::new(1.0, 0.0, 0.0, 0.0, 1.5_f64.sqrt(), 0.0);
    let stm0 = Matrix6::<f64>::identity();
    let tau = 2.0 * PI * 8.0_f64.sqrt(); // Orbit period.

    // Seed the dual state to recover sensitivities at the final solution.
    let y0_dual = Vector6::<DualSVec64<6>>::from_fn(|i, _| {
        DualSVec64::new(y0[i], Derivative::new(Some(stm0.row(i).transpose())))
    });

    // Augment the state with the initial STM.
    let y0_aug = stack![y0; SVector::<f64, 36>::from_iterator(stm0.iter().copied())];

    // Solve the system with dual numbers.
    let ode = TwoBodyODE {
        mu: DualSVec64::<6>::from_re(1.0),
    };
    let problem = ODEProblem::new(
        &ode,
        DualSVec64::<6>::from(0.0),
        DualSVec64::<6>::from(tau / 2.0),
        y0_dual,
    );
    let mut solver = ExplicitRungeKutta::dop853()
        .atol(DualSVec64::<6>::from(1e-14))
        .rtol(DualSVec64::<6>::from(1e-14));
    let sol = problem.solve(&mut solver).unwrap();

    // Recompute the same solution with a filtered step size.
    // Here the filter keeps only the real part of the dual step size so the
    // step length does not carry derivative information.
    solver = solver.filter(|h| DualSVec64::<6>::from(h.re()));
    let sol_flt = problem.solve(&mut solver).unwrap();

    let check = sol_flt
        .t
        .iter()
        .flat_map(|t| get_derivative::<6>(t))
        .map(|e| e.abs())
        .sum::<f64>();
    assert_eq!(check, 0.0);

    // Extract the state and STM at the penultimate step.
    let t1 = sol.t.last_chunk::<2>().unwrap()[0];
    let y1_dual = sol.y.last_chunk::<2>().unwrap()[0];
    let y1_eps: Vec<[f64; 6]> = y1_dual.iter().map(|d| get_derivative::<6>(d)).collect();

    let y1 = Vector6::<f64>::from_fn(|i, _| y1_dual[i].re());
    let stm1 = Matrix6::<f64>::from_fn(|r, c| y1_eps[r][c]);

    let t1_flt = sol_flt.t.last_chunk::<2>().unwrap()[0];
    assert_eq!(t1_flt.re(), t1.re());
    let y1_dual_flt = sol_flt.y.last_chunk::<2>().unwrap()[0];
    let y1_eps_flt: Vec<[f64; 6]> = y1_dual_flt.iter().map(|d| get_derivative::<6>(d)).collect();

    let y1_flt = Vector6::<f64>::from_fn(|i, _| y1_dual_flt[i].re());
    for (x, y) in zip(y1.as_slice(), y1_flt.as_slice()) {
        assert_eq!(x, y);
    }
    let stm1_flt = Matrix6::<f64>::from_fn(|r, c| y1_eps_flt[r][c]);

    // Solve the same problem with variational equations.
    let var_ode = VariationalTwoBodyODE { mu: 1.0 };
    let var_problem = ODEProblem::new(&var_ode, 0.0, t1.re(), y0_aug);
    let mut var_solver = ExplicitRungeKutta::dop853().atol(1e-14).rtol(1e-14);
    let var_sol = var_problem.solve(&mut var_solver).unwrap();

    // Extract the analytical STM from the augmented solution.
    let y1_aug = var_sol.last().unwrap().1;
    let y1_var = y1_aug.fixed_rows::<6>(0);
    let stm1_var = Matrix6::<f64>::from_iterator(y1_aug.fixed_rows::<36>(6).iter().copied());

    println!("Two-body state at {}: {}", t1, y1);
    println!("Filtered-step state at {}: {}", t1_flt, y1_flt);
    println!("Augmented-state solution at {}: {}", t1.re(), y1_var);
    println!("STM at {}: {}", t1, stm1);
    println!("Filtered-step STM at {}: {}", t1_flt, stm1_flt);
    println!("Variational STM at {}: {}", t1.re(), stm1_var);
}
