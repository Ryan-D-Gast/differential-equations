# Adjoint Sensitivity Analysis Design Proposal

This document outlines the proposed API for adding adjoint sensitivity analysis to the `differential-equations` library. This proposal builds upon the existing ODE solver infrastructure and the dual number support introduced in Issue 13.

## 1. Motivation

Adjoint methods are essential for:
- **Optimization**: Efficiently computing gradients of a scalar cost function with respect to many parameters.
- **Sensitivity Analysis**: Understanding how changes in initial conditions or parameters affect the final state.
- **Machine Learning**: Implementing Neural ODEs where the ODE solver is a layer in a larger differentiable model.

Compared to forward sensitivity analysis, adjoint methods are significantly more efficient when the number of parameters is large, as the gradient is computed in a single backward pass.

## 2. Core Components

### 2.1. The Objective Trait

To perform adjoint analysis, we need to define a scalar objective functional $L$:
$$L(p) = G(y(t_f), p) + \int_{t_0}^{t_f} g(t, y(t), p) \, dt$$

```rust
pub trait Objective<T: Real, Y: State<T>, P: State<T>> {
    /// Terminal cost G(y(tf), p)
    fn terminal_cost(&self, tf: T, yf: &Y, p: &P) -> T;

    /// Running cost (integrand) g(t, y(t), p)
    fn running_cost(&self, t: T, y: &Y, p: &P) -> T;

    /// Gradient of terminal cost w.r.t. state: ∂G/∂y
    fn dg_dy(&self, tf: T, yf: &Y, p: &P) -> Y;

    /// Gradient of terminal cost w.r.t. parameters: ∂G/∂p
    fn dg_dp(&self, tf: T, yf: &Y, p: &P) -> P;

    /// Partial derivative of running cost w.r.t. state: ∂g/∂y
    fn dr_dy(&self, t: T, y: &Y, p: &P) -> Y;

    /// Partial derivative of running cost w.r.t. parameters: ∂g/∂p
    fn dr_dp(&self, t: T, y: &Y, p: &P) -> P;
}
```

### 2.2. The Adjoint ODE Trait

The adjoint equation requires Vector-Jacobian Products (VJPs) to evolve the adjoint state $\lambda$:
$$\dot{\lambda}^T = -\lambda^T \frac{\partial f}{\partial y} - \frac{\partial g}{\partial y}$$

```rust
pub trait AdjointODE<T: Real, Y: State<T>, P: State<T>>: ODE<T, Y> {
    /// Vector-Jacobian Product with respect to state: v^T * (∂f/∂y)
    fn vjp_y(&self, t: T, y: &Y, p: &P, v: &Y) -> Y;

    /// Vector-Jacobian Product with respect to parameters: v^T * (∂f/∂p)
    fn vjp_p(&self, t: T, y: &Y, p: &P, v: &Y) -> P;
}
```

*Note: For many users, these VJPs can be automatically generated if the ODE is implemented using types that support Automatic Differentiation (AD).*

### 2.3. The Adjoint Problem

An `AdjointProblem` encapsulates everything needed to compute sensitivities.

```rust
pub struct AdjointProblem<'a, T, Y, P, F, L> {
    pub problem: ODEProblem<'a, T, Y, F>,
    pub objective: &'a L,
    pub p: P,
}

impl<'a, T, Y, P, F, L> AdjointProblem<'a, T, Y, P, F, L>
where
    T: Real, Y: State<T>, P: State<T>,
    F: AdjointODE<T, Y, P>,
    L: Objective<T, Y, P>
{
    pub fn solve_adjoint<S>(&self, solver: &mut S) -> Result<AdjointSolution<T, Y, P>, Error<T, Y>>
    where S: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y>
    {
        // 1. Forward pass: Solve the ODE and store the trajectory
        // 2. Backward pass: Solve the adjoint ODE from tf to t0
        // 3. Accumulate gradients
        unimplemented!()
    }
}
```

## 3. Trajectory Management

Continuous adjoint methods require the forward state $y(t)$ to compute Jacobians during the backward pass. We propose two strategies:

1.  **Interpolation (Dense Output)**: Store the full solution and use the existing `Interpolation` trait to query $y(t)$ at any time. This is easy to implement but can be memory-intensive for large systems or long time horizons.
2.  **Checkpointing**: Store the state at specific time points and re-integrate forward from the nearest checkpoint during the backward pass. This trades computation for memory.

## 4. Integration with Dual Numbers (Issue 13)

If a user implements their `ODE` using dual numbers, we can provide a blanket implementation of `AdjointODE`:

```rust
impl<F, T, Y, P> AdjointODE<T, Y, P> for F
where
    F: ODE<Dual<T>, DualVec<Y>>, // Pseudo-code for AD support
    ...
{
    fn vjp_y(&self, t: T, y: &Y, p: &P, v: &Y) -> Y {
        // Use AD to compute v^T * df/dy
    }
}
```

## 5. Example Usage (Mockup)

```rust
let ode = MySystem;
let objective = MyMSEObjective;
let p = vector![1.0, 0.5];

let problem = ODEProblem::new(&ode, 0.0, 10.0, y0);
let adjoint_problem = AdjointProblem::new(problem, &objective, p);

let mut solver = ExplicitRungeKutta::dop853();
let adj_sol = adjoint_problem.solve_adjoint(&mut solver).unwrap();

println!("Gradient w.r.t parameters: {:?}", adj_sol.grad_p);
println!("Gradient w.r.t initial condition: {:?}", adj_sol.grad_y0);
```

## 6. Implementation Roadmap

1.  **Enhance Solvers for Backward Integration**: Ensure that all solvers correctly handle `tf < t0` (this is mostly already supported).
2.  **Add Objective and Adjoint Traits**: Define the interfaces as described above.
3.  **Implement `AdjointProblem`**: Create the driver that runs the forward and backward passes.
4.  **Automatic VJP Support**: Integrate with a library like `num-dual` or provide internal AD helpers to simplify user implementation.
