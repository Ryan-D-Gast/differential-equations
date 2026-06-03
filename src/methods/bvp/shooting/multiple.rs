use crate::{
    bvp::Boundary,
    error::Error,
    interpolate::Interpolation,
    linalg::{Matrix, lin_solve, lu_decomp},
    methods::{ToleranceConfig, bvp::BVPMethod},
    ode::{ODE, OrdinaryNumericalMethod, solve_ode},
    solout::{DefaultSolout, Solout, TEvalSolout},
    solution::Solution,
    stats::{Evals, Steps},
    tolerance::Tolerance,
    traits::{Real, State},
};

/// Multiple-shooting method for ODE boundary value problems.
///
/// This method partitions the interval, solves an IVP on each subinterval, and
/// applies Newton iteration to enforce both endpoint boundary conditions and
/// continuity between neighboring shooting segments.
#[derive(Clone, Debug)]
pub struct MultipleShooting<M> {
    segments: usize,
    max_iterations: usize,
    tolerance: f64,
    ode_solver: M,
}

impl<M> MultipleShooting<M> {
    /// Create a multiple-shooting method from an ODE IVP solver.
    pub fn new(ode_solver: M) -> Self {
        Self {
            segments: 4,
            max_iterations: 100,
            tolerance: 1e-6,
            ode_solver,
        }
    }

    /// Set the number of shooting segments.
    pub fn segments(mut self, segments: usize) -> Self {
        self.segments = segments;
        self
    }

    /// Set the maximum number of Newton iterations.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the infinity-norm tolerance for the full multiple-shooting residual.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl<M, T> ToleranceConfig<T> for MultipleShooting<M>
where
    T: Real,
    M: ToleranceConfig<T>,
{
    fn rtol<V: Into<Tolerance<T>>>(mut self, rtol: V) -> Self {
        self.ode_solver = self.ode_solver.rtol(rtol);
        self
    }

    fn atol<V: Into<Tolerance<T>>>(mut self, atol: V) -> Self {
        self.ode_solver = self.ode_solver.atol(atol);
        self
    }
}

struct BvpToOde<'a, EqType: ?Sized> {
    problem: &'a EqType,
}

impl<EqType, T: Real, Y: State<T>> ODE<T, Y> for BvpToOde<'_, EqType>
where
    EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
{
    #[inline]
    fn diff(&self, t: T, y: &Y, dydt: &mut Y) {
        self.problem.diff(t, y, dydt);
    }
}

impl<M> MultipleShooting<M> {
    fn mesh<T, Y>(&self, t0: T, tf: T) -> Result<Vec<T>, Error<T, Y>>
    where
        T: Real,
        Y: State<T>,
    {
        if self.segments == 0 {
            return Err(Error::BadInput {
                msg: "Multiple shooting requires at least one segment.".to_string(),
            });
        }

        let mut mesh = Vec::with_capacity(self.segments + 1);
        let span = tf - t0;
        let denominator = T::from_usize(self.segments).ok_or_else(|| Error::BadInput {
            msg: "Could not represent multiple-shooting segment count as scalar type.".to_string(),
        })?;

        for i in 0..=self.segments {
            let numerator = T::from_usize(i).ok_or_else(|| Error::BadInput {
                msg: "Could not represent multiple-shooting mesh index as scalar type.".to_string(),
            })?;
            mesh.push(t0 + span * numerator / denominator);
        }

        Ok(mesh)
    }

    fn initial_nodes<EqType, T, Y>(
        &self,
        ode_system: &BvpToOde<'_, EqType>,
        mesh: &[T],
        t0: T,
        tf: T,
        y_guess: &Y,
    ) -> Result<(Vec<Y>, Evals, Steps), Error<T, Y>>
    where
        T: Real,
        Y: State<T>,
        M: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y> + Clone,
        EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
    {
        let mut solver = self.ode_solver.clone();
        let mut solout = TEvalSolout::new(mesh, t0, tf);
        let solution = solve_ode(&mut solver, ode_system, t0, tf, y_guess, &mut solout)?;
        if solution.y.len() != mesh.len() {
            return Err(Error::BadInput {
                msg: "Initial multiple-shooting IVP did not produce every mesh node.".to_string(),
            });
        }

        Ok((solution.y, solution.evals, solution.steps))
    }

    fn residual<EqType, T, Y>(
        &self,
        problem: &EqType,
        ode_system: &BvpToOde<'_, EqType>,
        mesh: &[T],
        nodes: &[Y],
        dim: usize,
    ) -> Result<(Vec<T>, Evals, Steps), Error<T, Y>>
    where
        T: Real,
        Y: State<T>,
        M: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y> + Clone,
        EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
    {
        let mut residual = Vec::with_capacity(nodes.len() * dim);
        let mut boundary_residual = nodes[0].zeros_like();
        let mut total_evals = Evals::new();
        let mut total_steps = Steps::new();

        problem.boundary(&nodes[0], &nodes[nodes.len() - 1], &mut boundary_residual);
        for i in 0..dim {
            residual.push(boundary_residual.get_component(i));
        }

        for segment_idx in 0..self.segments {
            let mut solver = self.ode_solver.clone();
            let mut solout = DefaultSolout::new();
            let solution = solve_ode(
                &mut solver,
                ode_system,
                mesh[segment_idx],
                mesh[segment_idx + 1],
                &nodes[segment_idx],
                &mut solout,
            )?;
            total_evals += solution.evals;
            total_steps += solution.steps;

            let (_, y_end) = solution.last().map_err(|err| Error::BadInput {
                msg: format!("Internal multiple-shooting IVP returned an empty solution: {err}"),
            })?;

            for i in 0..dim {
                residual.push(y_end.get_component(i) - nodes[segment_idx + 1].get_component(i));
            }
        }

        Ok((residual, total_evals, total_steps))
    }

    fn residual_norm<T: Real>(residual: &[T]) -> T {
        residual
            .iter()
            .fold(T::zero(), |max_norm, value| max_norm.max(value.abs()))
    }

    fn apply_newton_step<T, Y>(nodes: &mut [Y], step: &[T], dim: usize)
    where
        T: Real,
        Y: State<T>,
    {
        for (node_idx, node) in nodes.iter_mut().enumerate() {
            for component_idx in 0..dim {
                let index = node_idx * dim + component_idx;
                node.set_component(
                    component_idx,
                    node.get_component(component_idx) + step[index],
                );
            }
        }
    }
}

impl<M, T, Y> BVPMethod<T, Y> for MultipleShooting<M>
where
    T: Real,
    Y: State<T>,
    M: OrdinaryNumericalMethod<T, Y> + Interpolation<T, Y> + Clone,
{
    fn solve<EqType, SoloutType>(
        &mut self,
        problem: &EqType,
        t0: T,
        tf: T,
        y_guess: &Y,
        solout: &mut SoloutType,
    ) -> Result<Solution<T, Y>, Error<T, Y>>
    where
        EqType: ODE<T, Y> + Boundary<T, Y> + ?Sized,
        SoloutType: Solout<T, Y>,
    {
        let dim = y_guess.len();
        let unknowns = (self.segments + 1) * dim;
        let mesh = self.mesh(t0, tf)?;
        let ode_system = BvpToOde { problem };
        let tolerance = T::from_f64(self.tolerance).ok_or_else(|| Error::BadInput {
            msg: "BVP multiple-shooting tolerance cannot be represented by scalar type."
                .to_string(),
        })?;
        let eps = T::default_epsilon().sqrt();

        let (mut nodes, evals, steps) = self.initial_nodes(&ode_system, &mesh, t0, tf, y_guess)?;
        let mut total_evals = evals;
        let mut total_steps = steps;
        let mut jacobian = Matrix::<T>::zeros(unknowns, unknowns);
        let mut ip = vec![0; unknowns];

        for _ in 0..self.max_iterations {
            let (residual, evals, steps) =
                self.residual(problem, &ode_system, &mesh, &nodes, dim)?;
            total_evals += evals;
            total_steps += steps;

            if Self::residual_norm(&residual) <= tolerance {
                let mut final_solver = self.ode_solver.clone();
                let mut solution =
                    solve_ode(&mut final_solver, &ode_system, t0, tf, &nodes[0], solout)?;
                solution.evals += total_evals;
                solution.steps += total_steps;
                return Ok(solution);
            }

            for j in 0..unknowns {
                let node_idx = j / dim;
                let component_idx = j % dim;
                let mut perturbed_nodes = nodes.clone();
                let y_j = perturbed_nodes[node_idx].get_component(component_idx);
                let perturbation = eps * y_j.abs().max(T::one());
                perturbed_nodes[node_idx].set_component(component_idx, y_j + perturbation);

                let (perturbed_residual, evals, steps) =
                    self.residual(problem, &ode_system, &mesh, &perturbed_nodes, dim)?;
                total_evals += evals;
                total_steps += steps;
                total_evals.jacobian += 1;

                for i in 0..unknowns {
                    jacobian[(i, j)] = (perturbed_residual[i] - residual[i]) / perturbation;
                }
            }

            let mut step = residual.iter().map(|value| -*value).collect::<Vec<_>>();
            lu_decomp(&mut jacobian, &mut ip).map_err(|err| Error::LinearAlgebra {
                t: t0,
                y: nodes[0].clone(),
                msg: err.to_string(),
            })?;
            lin_solve(&jacobian, &mut step, &ip);
            total_evals.newton += 1;
            total_evals.decompositions += 1;
            total_evals.solves += 1;
            Self::apply_newton_step(&mut nodes, &step, dim);
        }

        Err(Error::MaxSteps {
            t: t0,
            y: nodes.first().cloned().unwrap_or_else(|| y_guess.clone()),
        })
    }
}
