# Differential-Algebraic Equations (DAE)

The `dae` module provides tools for solving differential-algebraic equations (DAEs), focusing on initial value problems (DAEProblems) and supporting index-1, index-2, and index-3 systems.

## Table of Contents
- [Defining a DAE](#defining-a-dae)
- [Solving an Initial Value Problem (DAEProblem)](#solving-an-initial-value-problem)
- [Examples](#examples)

## Defining a DAE

The `DAE` trait defines the system in the form `M·y' = f(t, y)`, where `M` is the mass matrix and `f` is the right-hand side. The trait also supports an optional `event` function to interrupt the solver when a condition is met.

### DAE Trait
* `diff` - Right-hand side function `f(t, &y, &mut f)`.
* `mass` - Mass matrix function `mass(&mut M)`.
* `event` - Optional event function to interrupt the solver when a condition is met, returning `ControlFlag::Terminate(reason: CallBackData)`.

### Solout Trait
* `solout` - Function to choose which points to save in the solution. Common implementations are included in the `solout` module. The `DAEProblem` trait implements methods to use them easily, e.g. `even`, `dense`, and `t_eval`.

### Implementation (Amplifier DAE example)

The amplifier example from `examples/dae/01_amplifier/main.rs` demonstrates an 8-dimensional index-1 DAE with a singular mass matrix and nonlinear (exponential) elements. Below is a compact excerpt that illustrates how to implement `diff` and `mass` for the amplifier model.

```rust
use differential_equations::prelude::*;
use nalgebra::{SVector, vector};

/// The full example lives in `examples/dae/01_amplifier/main.rs`.
struct AmplifierModel {
    ue: f64,
    ub: f64,
    uf: f64,
    alpha: f64,
    beta: f64,
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
        let w = 2.0 * std::f64::consts::PI * 100.0;
        let uet = self.ue * (w * t).sin();

        let fac1 = self.beta * (((y[3] - y[2]) / self.uf).exp() - 1.0);
        let fac2 = self.beta * (((y[6] - y[5]) / self.uf).exp() - 1.0);

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
        // Main diagonal
        m[(0, 0)] = -self.c5;
        m[(1, 1)] = -self.c5;
        m[(2, 2)] = -self.c4;
        m[(3, 3)] = -self.c3;
        m[(4, 4)] = -self.c3;
        m[(5, 5)] = -self.c2;
        m[(6, 6)] = -self.c1;
        m[(7, 7)] = -self.c1;

        // Super diagonal
        m[(0, 1)] = self.c5;
        m[(3, 4)] = self.c3;
        m[(6, 7)] = self.c1;

        // Sub diagonal
        m[(1, 0)] = self.c5;
        m[(4, 3)] = self.c3;
        m[(7, 6)] = self.c1;
    }

    fn jacobian(&self, _t: f64, y: &SVector<f64, 8>, jac: &mut Matrix<f64>) {
        // Sensitivities of exponential terms
        let g14 = self.beta * ((y[3] - y[2]) / self.uf).exp() / self.uf;
        let g27 = self.beta * ((y[6] - y[5]) / self.uf).exp() / self.uf;

        // Jacobian Matrix
        jac[(0, 0)] = 1.0 / self.r9;
        jac[(1, 1)] = 1.0 / self.r8;
        jac[(1, 3)] = self.alpha * g14;
        jac[(1, 2)] = -self.alpha * g14;
        jac[(2, 2)] = 1.0 / self.r7 + g14;
        jac[(2, 3)] = -g14;
        jac[(3, 3)] = 1.0 / self.r5 + 1.0 / self.r6 + (1.0 - self.alpha) * g14;
        jac[(3, 2)] = -(1.0 - self.alpha) * g14;
        jac[(4, 4)] = 1.0 / self.r4;
        jac[(4, 6)] = self.alpha * g27;
        jac[(4, 5)] = -self.alpha * g27;
        jac[(5, 5)] = 1.0 / self.r3 + g27;
        jac[(5, 6)] = -g27;
        jac[(6, 6)] = 1.0 / self.r1 + 1.0 / self.r2 + (1.0 - self.alpha) * g27;
        jac[(6, 5)] = -(1.0 - self.alpha) * g27;
        jac[(7, 7)] = 1.0 / self.r0;
    }
}
```

## Solving an Initial Value Problem

The `DAEProblem` trait is used to solve the system using the solver. The trait includes methods to set the initial conditions, solve the system, and get the solution. The `solve` method returns a `Result<Solution, Status>` where `Solution` contains the solution and statistics.

```rust
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
    let t0 = 0.0;
    let tf = 0.05;

    let problem = DAEProblem::new(model, t0, tf, y0);
    match problem.even(0.0025).solve(&mut method) {
        Ok(solution) => {
            println!("Solution:");
            for (t, y) in solution.iter() {
                println!("{:.4} {:?}", t, y);
            }
            println!("Function evaluations: {}", solution.evals.function);
        }
        Err(e) => panic!("Error: {:?}", e),
    }
}
```

## Examples

For more examples, see the `examples` directory. The examples demonstrate different systems, methods, and output methods for different use cases.

| Example | Description & Demonstrated Features |
|---|---|
| [Amplifier DAE](../../examples/dae/01_amplifier/main.rs) | Simple index-1 DAE for an amplifier circuit. |
| [Robertson DAE](../../examples/dae/02_robertson/main.rs) | Stiff chemical kinetics DAE with conservation constraint. |

### Amplifier DAE Example Output

```
Amplifier DAE Solution:
Time     Y[0]              Y[1]              NSTEP
 0.0000     0.0000000000e0     6.0000000000e0      0
 0.0025    4.3516991222e-3     6.0045583009e0      1
 0.0050    4.2278512534e-3     6.0046728642e0      2
 0.0075    4.1165476872e-3     6.0047930921e0      3
 0.0100    4.0045517143e-3     6.0049066607e0      4
 0.0125    -2.5259407749e0     3.4634019125e0      5
 0.0150    -1.7525423226e0     4.0560492607e0      6
 0.0175    1.0822010043e-1     5.8946214149e0      7
 0.0200    1.0814698817e-1     5.9006241065e0      8
 0.0225    -4.2534133306e0     1.4541357784e0      9
 0.0250    -1.6167649413e0     3.9071049103e0     10
 0.0275    2.4666792313e-1     5.7559985011e0     11
 0.0300    2.4277052064e-1     5.7657672155e0     12
 0.0325    -4.1294779779e0     1.3150897876e0     13
 0.0350    -1.4939802105e0     3.7737789902e0     14
 0.0375    3.7115418728e-1     5.6312596348e0     15
 0.0400    3.6364762672e-1     5.6442455894e0     16
 0.0425    -4.0171912750e0     1.1908843667e0     17
 0.0450    -1.3829565815e0     3.6545229219e0     18
 0.0475    4.8339787186e-1     5.5194284082e0     19
 0.0500    4.7269794728e-1     5.5353588484e0     20

Final solution at t =  0.0500:
  Y[0] =    4.7269794728e-1
  Y[1] =     5.5353588484e0

Solver Statistics:
  Function evaluations: 1844
  Jacobian evaluations: 123
  LU decompositions: 187
  Linear solves: 1271
  Successful steps: 124
  Rejected steps: 63
  Total steps: 187
  Solve time: 0.0372095
```