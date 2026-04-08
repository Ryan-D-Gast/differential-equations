use differential_equations::{
    ode::{ODE, ODEProblem},
    prelude::ExplicitRungeKutta,
    traits::Real,
};
use nalgebra::{Dim, Matrix3, Matrix6, SVector, Vector6, stack};
use num_dual::{Derivative, DualSVec64, DualStruct};
use std::{f64::consts::PI, iter::zip};

/// Ordinary differential equation for the Keplerian dynamics
struct KeplerODE<T: Real> {
    /// gravitational parameter
    mu: T,
}

impl<T> ODE<T, Vector6<T>> for KeplerODE<T>
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

/// Variational equations for the Keplerian dynamics
struct VariationalKeplerODE {
    /// gravitational parameter
    mu: f64,
}

impl ODE<f64, SVector<f64, 42>> for VariationalKeplerODE {
    fn diff(&self, _: f64, y: &SVector<f64, 42>, dydt: &mut SVector<f64, 42>) {
        let r = y.fixed_rows::<3>(0);
        let c = self.mu / r.norm().powi(3);

        // time derivatives of the six-dimensional state
        dydt.fixed_rows_mut::<3>(0).copy_from(&y.fixed_rows::<3>(3));
        dydt.fixed_rows_mut::<3>(3).copy_from(&(r * (-c)));

        // time derivatives of the 6x6 state transition matrix
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
    // initial state: periapis of an elliptical orbit with periapsis radius equal to 1.0 and eccentricity equal to 0.5
    let y0 = Vector6::<f64>::new(1.0, 0.0, 0.0, 0.0, 1.5_f64.sqrt(), 0.0);
    let stm0 = Matrix6::<f64>::identity(); // initial state transition matrix
    let tau = 2.0 * PI * 8.0_f64.sqrt(); // orbit period

    // initial state seeded to compute the partials of the flow of the solution w.r.t. the initial conditions
    let y0_dual = Vector6::<DualSVec64<6>>::from_fn(|i, _| {
        DualSVec64::new(y0[i], Derivative::new(Some(stm0.row(i).transpose())))
    });

    // initial state augmented with the state transition matrix at the initial time
    let y0_aug = stack![y0; SVector::<f64, 36>::from_iterator(stm0.iter().copied())];

    // compute the flow of the solution to the Kepler equation and the partials
    // w.r.t. the initial conditions using automatic differentiation
    let ode = KeplerODE {
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

    // recompute the same solution but filtering the step size such that its derivatives are identically zero
    solver = solver.filter(|h| DualSVec64::<6>::from(h.re()));
    let sol_flt = problem.solve(&mut solver).unwrap();

    let check = sol_flt
        .t
        .iter()
        .flat_map(|t| get_derivative::<6>(t))
        .map(|e| e.abs())
        .sum::<f64>();
    assert_eq!(check, 0.0);

    // extract the state and state transition matrix at the penultimate step
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

    // compute the flow of the solution to the Kepler equation and the
    // partials w.r.t. the initial conditions via variational equations
    let var_ode = VariationalKeplerODE { mu: 1.0 };
    let var_problem = ODEProblem::new(&var_ode, 0.0, t1.re(), y0_aug);
    let mut var_solver = ExplicitRungeKutta::dop853().atol(1e-14).rtol(1e-14);
    let var_sol = var_problem.solve(&mut var_solver).unwrap();

    // extract the state and analytical state transition matrix
    let y1_aug = var_sol.last().unwrap().1;
    let y1_var = y1_aug.fixed_rows::<6>(0);
    let stm1_var = Matrix6::<f64>::from_iterator(y1_aug.fixed_rows::<36>(6).iter().copied());

    println!("State at {}: {}", t1, y1);
    println!("State with filtered step at {}: {}", t1_flt, y1_flt);
    println!("State from augmented solution at {}: {}", t1.re(), y1_var);
    println!("State transition matrix at {}: {}", t1, stm1);
    println!(
        "State transition matrix with filtered step at {}: {}",
        t1_flt, stm1_flt
    );
    println!(
        "Analytical state transition matrix at {}: {}",
        t1.re(),
        stm1_var
    );
}
