//! Example 02: Robertson Chemical Kinetics DAE
//!
//! This example models the Robertson chemical reaction system, which is a classic
//! benchmark problem for stiff DAE solvers. The system represents a simple
//! autocatalytic chemical reaction with three species A, B, and C.
//!
//! The chemical reactions are:
//! A → B         (rate k₁)
//! B + B → C + B (rate k₂)  
//! B + C → A + C (rate k₃)
//!
//! The DAE system is:
//! dy₁/dt = -k₁*y₁ + k₃*y₂*y₃
//! dy₂/dt = k₁*y₁ - k₃*y₂*y₃ - k₂*y₂²
//! 0 = y₁ + y₂ + y₃ - 1  (conservation constraint)
//!
//! This is an index-1 DAE with a mass matrix:
//! M = [1 0 0]
//!     [0 1 0]
//!     [0 0 0]
//!
//! The system is highly stiff with widely separated time scales, making it
//! an excellent test case for implicit DAE solvers.

use differential_equations::prelude::*;
use nalgebra::{SVector, vector};
use quill::prelude::*;

/// Robertson Chemical Kinetics DAE Model
struct RobertsonModel {
    k1: f64, // Rate constant for A → B
    k2: f64, // Rate constant for B + B → C + B
    k3: f64, // Rate constant for B + C → A + C
}

impl RobertsonModel {
    fn new(k1: f64, k2: f64, k3: f64) -> Self {
        Self { k1, k2, k3 }
    }
}

// State vector: [y1, y2, y3] representing concentrations of species A, B, C
impl DAE<f64, SVector<f64, 3>> for RobertsonModel {
    fn diff(&self, _t: f64, y: &SVector<f64, 3>, f: &mut SVector<f64, 3>) {
        let y1 = y[0]; // Concentration of A
        let y2 = y[1]; // Concentration of B
        let y3 = y[2]; // Concentration of C

        // Chemical kinetics equations
        f[0] = -self.k1 * y1 + self.k3 * y2 * y3;
        f[1] = self.k1 * y1 - self.k3 * y2 * y3 - self.k2 * y2 * y2;
        f[2] = y1 + y2 + y3 - 1.0; // Conservation constraint
    }

    fn mass(&self, m: &mut Matrix<f64>) {
        // Mass matrix for Robertson DAE
        m[(0, 0)] = 1.0; // dy1/dt equation (differential)
        m[(1, 1)] = 1.0; // dy2/dt equation (differential)
        m[(2, 2)] = 0.0; // Conservation constraint (algebraic)
    }

    fn jacobian(&self, _t: f64, y: &SVector<f64, 3>, j: &mut Matrix<f64>) {
        let y2 = y[1];
        let y3 = y[2];

        // Row 0: dF1/dy
        j[(0, 0)] = -self.k1;
        j[(0, 1)] = self.k3 * y3;
        j[(0, 2)] = self.k3 * y2;

        // Row 1: dF2/dy
        j[(1, 0)] = self.k1;
        j[(1, 1)] = -self.k3 * y3 - 2.0 * self.k2 * y2;
        j[(1, 2)] = -self.k3 * y2;

        // Row 2: dF3/dy for algebraic constraint y1 + y2 + y3 - 1 = 0
        j[(2, 0)] = 1.0;
        j[(2, 1)] = 1.0;
        j[(2, 2)] = 1.0;
    }
}

fn main() {
    // DAE solver for stiff chemical kinetics
    let mut method = ImplicitRungeKutta::radau5()
        .rtol(1e-5)
        .atol([1e-6, 1e-10, 1e6]);

    // Robertson reaction rate constants
    let k1 = 0.04; // A → B
    let k2 = 3.0e7; // B + B → C + B (fast reaction)
    let k3 = 1.0e4; // B + C → A + C

    let model = RobertsonModel::new(k1, k2, k3);

    // Initial conditions: all A, no B or C
    let y0 = vector![
        1.0, // y1 = A concentration
        0.0, // y2 = B concentration
        0.0, // y3 = C concentration
    ];

    // Simulation time - very long time span to see equilibrium
    let t0 = 0.0;
    let tf = 4.0 * 10f64.powf(6.0);

    let robertson_problem = DAEProblem::new(model, t0, tf, y0);

    // Output points: ~200 log-spaced points between 1e-6 and 1e6
    let n_pts = 200usize;
    let exp_min = -6.0f64;
    let exp_max = 6.0f64;
    let step = (exp_max - exp_min) / ((n_pts - 1) as f64);
    let points = (0..n_pts)
        .map(|i| 4.0f64 * 10f64.powf(exp_min + (i as f64) * step))
        .collect::<Vec<_>>();

    // Solve the DAE
    match robertson_problem.t_eval(points).solve(&mut method) {
        Ok(solution) => {
            // Print the statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);

            println!("\nRobertson DAE Solution:");
            println!("{:>12}  {:>12}   {:>12}   {:>12}", "Time", "A", "B", "C");
            for (t, y) in solution.iter() {
                println!("{:12.4}  {:12.8}   {:12.8}   {:12.8}", t, y[0], y[1], y[2]);
            }

            // Create semilog-x plot with three series: y[:,0], 1e4*y[:,1], y[:,2]
            // Skip t == 0 because log(0) is undefined.
            let mut s1: Vec<(f64, f64)> = Vec::new();
            let mut s2: Vec<(f64, f64)> = Vec::new();
            let mut s3: Vec<(f64, f64)> = Vec::new();
            for (t, y) in solution.iter() {
                let tv = *t;
                if tv > 0.0 {
                    s1.push((tv, y[0]));
                    s2.push((tv, 1e4 * y[1]));
                    s3.push((tv, y[2]));
                }
            }

            Plot::builder()
                .title("Robertson DAE problem with a Conservation Law")
                .x_label("Time (t)")
                .y_label("y1, 1e4 * y2, y3")
                .x_scale(Scale::Log)
                .legend(Legend::TopLeftInside)
                .data([
                    Series::builder()
                        .name("Concentration A")
                        .color("Blue")
                        .data(s1)
                        .build(),
                    Series::builder()
                        .name("Concentration B")
                        .color("Orange")
                        .data(s2)
                        .build(),
                    Series::builder()
                        .name("Concentration C")
                        .color("Green")
                        .data(s3)
                        .build(),
                ])
                .build()
                .to_svg("examples/dae/02_robertson/robertson.svg")
                .expect("Failed to save Robertson semilogx plot as SVG");
        }
        Err(e) => panic!("Error solving Robertson DAE: {:?}", e),
    }
}

/*
Output from FORTRAN Version
S C:\Users\Ryan\Desktop\Code\Rust\differential-equations\examples\dae\01_robertson> .\rob_radau5.exe
 X = 1.0E+05    Y =  0.1786592545E-01  0.7274753875E-07  0.9821340018E+00
       tol=0.10D-04
 fcn=  699 jac=  81 step=  91 accpt=  90 rejct=  1 dec=  91 sol=  203
*/
