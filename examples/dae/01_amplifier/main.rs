//! Example 01: Amplifier DAE Problem
//!
//! This example models an electrical amplifier circuit using a differential algebraic equation (DAE)
//! system. The problem comes from Hairer & Wanner's "Solving Ordinary Differential Equations II"
//! and is a classic test case for DAE solvers.
//!
//! The DAE system is in the form:
//! M * y' = f(t, y)
//!
//! where:
//! - M is a singular mass matrix representing the circuit topology
//! - y is the 8-dimensional state vector (node voltages and currents)
//! - f(t, y) contains the electrical circuit equations
//!
//! The amplifier circuit consists of:
//! - Capacitors that introduce differential equations (time derivatives)
//! - Resistors that introduce algebraic constraints (no time derivatives)
//! - A sinusoidal input voltage: u_e(t) = U_E * sin(2π * 100 * t)
//! - Nonlinear elements (diodes) with exponential I-V characteristics
//!
//! This example demonstrates:
//! - Using the mass matrix formulation for DAE systems
//! - Implementing nonlinear circuit equations with exponential terms
//! - Working with singular mass matrices (index-1 DAE)
//! - Using implicit Runge-Kutta methods for stiff DAE problems

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

/// Amplifier DAE Model
struct AmplifierModel {
    // Circuit parameters
    ue: f64,    // Input voltage amplitude
    ub: f64,    // Supply voltage
    uf: f64,    // Thermal voltage
    alpha: f64, // Current gain
    beta: f64,  // Saturation current
    // Resistor values
    r0: f64,
    r1: f64,
    r2: f64,
    r3: f64,
    r4: f64,
    r5: f64,
    r6: f64,
    r7: f64,
    r8: f64,
    r9: f64,
    // Capacitor values
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    c5: f64,
}

impl AmplifierModel {
    fn new() -> Self {
        Self {
            ue: 0.1,
            ub: 6.0,
            uf: 0.026,
            alpha: 0.99,
            beta: 1.0e-6,
            r0: 1000.0,
            r1: 9000.0,
            r2: 9000.0,
            r3: 9000.0,
            r4: 9000.0,
            r5: 9000.0,
            r6: 9000.0,
            r7: 9000.0,
            r8: 9000.0,
            r9: 9000.0,
            c1: 1.0e-6,
            c2: 2.0e-6,
            c3: 3.0e-6,
            c4: 4.0e-6,
            c5: 5.0e-6,
        }
    }
}

impl DAE<f64, SVector<f64, 8>> for AmplifierModel {
    fn diff(&self, t: f64, y: &SVector<f64, 8>, f: &mut SVector<f64, 8>) {
        // Sinusoidal input voltage
        let w = 2.0 * std::f64::consts::PI * 100.0;
        let uet = self.ue * (w * t).sin();

        // Nonlinear exponential terms (diode characteristics)
        let fac1 = self.beta * (((y[3] - y[2]) / self.uf).exp() - 1.0);
        let fac2 = self.beta * (((y[6] - y[5]) / self.uf).exp() - 1.0);

        // Circuit equations
        f[0] = y[0] / self.r9;
        f[1] = (y[1] - self.ub) / self.r8 + self.alpha * fac1;
        f[2] = y[2] / self.r7 - fac1;
        f[3] = y[3] / self.r5 + (y[3] - self.ub) / self.r6 + (1.0 - self.alpha) * fac1;
        f[4] = (y[4] - self.ub) / self.r4 + self.alpha * fac2;
        f[5] = y[5] / self.r3 - fac2;
        f[6] = y[6] / self.r1 + (y[6] - self.ub) / self.r2 + (1.0 - self.alpha) * fac2;
        f[7] = (y[7] - uet) / self.r0;
    }

    fn mass(&self, m: &mut Matrix<f64>) {
        // Main diagonal elements: B(2,j) -> m[(j-1,j-1)]
        m[(0, 0)] = -self.c5;
        m[(1, 1)] = -self.c5;
        m[(2, 2)] = -self.c4;
        m[(3, 3)] = -self.c3;
        m[(4, 4)] = -self.c3;
        m[(5, 5)] = -self.c2;
        m[(6, 6)] = -self.c1;
        m[(7, 7)] = -self.c1;

        // Super-diagonal elements: B(1,j) -> m[(j-2,j-1)]
        m[(0, 1)] = self.c5;
        m[(3, 4)] = self.c3;
        m[(6, 7)] = self.c1;

        // Sub-diagonal elements: B(3,j) -> m[(j,j-1)]
        m[(1, 0)] = self.c5;
        m[(4, 3)] = self.c3;
        m[(7, 6)] = self.c1;
    }

    fn jacobian(&self, _t: f64, y: &SVector<f64, 8>, jac: &mut Matrix<f64>) {
        // Sensitivities of exponential terms
        let g14 = self.beta * ((y[3] - y[2]) / self.uf).exp() / self.uf;
        let g27 = self.beta * ((y[6] - y[5]) / self.uf).exp() / self.uf;

        // df0/dy
        jac[(0, 0)] = 1.0 / self.r9;

        // df1/dy
        jac[(1, 1)] = 1.0 / self.r8;
        jac[(1, 3)] = self.alpha * g14; // d f1 / d y3
        jac[(1, 2)] = -self.alpha * g14; // d f1 / d y2

        // df2/dy
        jac[(2, 2)] = 1.0 / self.r7 + g14;
        jac[(2, 3)] = -g14;

        // df3/dy
        jac[(3, 3)] = 1.0 / self.r5 + 1.0 / self.r6 + (1.0 - self.alpha) * g14;
        jac[(3, 2)] = -(1.0 - self.alpha) * g14;

        // df4/dy
        jac[(4, 4)] = 1.0 / self.r4;
        jac[(4, 6)] = self.alpha * g27; // d f4 / d y6
        jac[(4, 5)] = -self.alpha * g27; // d f4 / d y5

        // df5/dy
        jac[(5, 5)] = 1.0 / self.r3 + g27;
        jac[(5, 6)] = -g27;

        // df6/dy
        jac[(6, 6)] = 1.0 / self.r1 + 1.0 / self.r2 + (1.0 - self.alpha) * g27;
        jac[(6, 5)] = -(1.0 - self.alpha) * g27;

        // df7/dy
        jac[(7, 7)] = 1.0 / self.r0;
    }
}

fn main() {
    // DAE solver with high accuracy for stiff problems
    let mut method = ImplicitRungeKutta::radau5()
        .rtol(1.0e-5)
        .atol(1.0e-11)
        .h0(1.0e-6);

    // Circuit model
    let model = AmplifierModel::new();

    // Initial conditions (computed from circuit steady-state)
    let y0 = vector![
        0.0,
        model.ub - 0.0 * model.r8 / model.r9,
        model.ub / (model.r6 / model.r5 + 1.0),
        model.ub / (model.r6 / model.r5 + 1.0),
        model.ub,
        model.ub / (model.r2 / model.r1 + 1.0),
        model.ub / (model.r2 / model.r1 + 1.0),
        0.0,
    ];

    // Simulation time
    let t0 = 0.0;
    let tf = 0.05; // 50 milliseconds

    let amplifier_problem = DAEProblem::new(model, t0, tf, y0);

    // Solve the DAE with output at regular intervals
    match amplifier_problem.even(0.0025).solve(&mut method) {
        Ok(solution) => {
            // Print the solution
            println!("Amplifier DAE Solution:");
            println!("Time     Y[0]              Y[1]              NSTEP");
            for (i, (t, y)) in solution.iter().enumerate() {
                println!("{:7.4}   {:16.10e}   {:16.10e}   {:4}", t, y[0], y[1], i);
            }

            // Print final result
            let (tf, yf) = solution.last().unwrap();
            println!("\nFinal solution at t = {:7.4}:", tf);
            println!("  Y[0] = {:18.10e}", yf[0]);
            println!("  Y[1] = {:18.10e}", yf[1]);

            // Print solver statistics
            println!("\nSolver Statistics:");
            println!("  Function evaluations: {}", solution.evals.function);
            println!("  Jacobian evaluations: {}", solution.evals.jacobian);
            println!("  LU decompositions: {}", solution.evals.decompositions);
            println!("  Linear solves: {}", solution.evals.solves);
            println!("  Successful steps: {}", solution.steps.accepted);
            println!("  Rejected steps: {}", solution.steps.rejected);
            println!("  Total steps: {}", solution.steps.total());
            println!("  Solve time: {:?}", solution.timer.elapsed());
        }
        Err(e) => panic!("Error solving DAE: {:?}", e),
    }
}

/*
Fortran output
X = 0.0275    Y =  0.2467076493E+00  0.5756038700E+01    NSTEP = 124
 X = 0.0300    Y =  0.2426560786E+00  0.5765652267E+01    NSTEP = 128
 X = 0.0325    Y = -0.4129386794E+01  0.1315181672E+01    NSTEP = 145
 X = 0.0350    Y = -0.1493819637E+01  0.3773941357E+01    NSTEP = 156
 X = 0.0375    Y =  0.3712447894E+00  0.5631351277E+01    NSTEP = 172
 X = 0.0400    Y =  0.3637331115E+00  0.5644330299E+01    NSTEP = 176
 X = 0.0425    Y = -0.4017313718E+01  0.1190761752E+01    NSTEP = 192
 X = 0.0450    Y = -0.1382850300E+01  0.3654630903E+01    NSTEP = 203
 X = 0.0475    Y =  0.4832264317E+00  0.5519257543E+01    NSTEP = 218
 X = 0.0500    Y =  0.4726948107E+00  0.5535355822E+01
       tol=0.10D-04
 fcn= 2596 jac= 214 step= 276 accpt= 221 rejct= 18 dec= 274 sol=  791
*/
