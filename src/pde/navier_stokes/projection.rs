//! Projection backend for 2D incompressible velocity fields.

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    ode::ODE,
    pde::{
        BoundaryCondition, BoundaryConditions, BoundaryFace, PDE, Side, SpatialDiscretization,
        StructuredGrid,
    },
    traits::{Real, State},
};

/// Projection-method spatial backend for two-dimensional incompressible flow.
///
/// The global state is laid out as `[u0, v0, u1, v1, ...]` with two velocity
/// components per grid node. The wrapped [`PDE`] supplies the unprojected
/// velocity tendency through its flux and source terms. This backend then solves
/// a pressure-like Poisson equation and subtracts its gradient so the returned
/// velocity tendency has substantially lower discrete divergence.
#[derive(Clone, Debug)]
pub struct ProjectionMethod<T, U = Vec<T>>
where
    T: Real,
    U: State<T>,
{
    grid: StructuredGrid<T, 2>,
    boundary: BoundaryConditions<T, U>,
    local_template: U,
}

impl<T> ProjectionMethod<T, Vec<T>>
where
    T: Real,
{
    /// Create a projection backend for a 2D velocity field on a structured grid.
    pub fn uniform(grid: StructuredGrid<T, 2>) -> Self {
        Self::with_field(grid, vec![T::zero(); 2])
    }
}

impl<T, U> ProjectionMethod<T, U>
where
    T: Real,
    U: State<T>,
{
    /// Create a projection backend with an explicit local velocity template.
    pub fn with_field(grid: StructuredGrid<T, 2>, local_template: U) -> Self {
        assert_eq!(
            local_template.len(),
            2,
            "ProjectionMethod expects two local velocity components"
        );
        let boundary = BoundaryConditions::homogeneous_neumann_like::<2>(&local_template);
        Self {
            grid,
            boundary,
            local_template,
        }
    }

    /// Set velocity boundary conditions.
    pub fn boundary(mut self, boundary: BoundaryConditions<T, U>) -> Self {
        self.boundary = boundary;
        self
    }
}

impl<'a, Eq, T, U, Y> SpatialDiscretization<'a, Eq, T, U, Y, 2> for ProjectionMethod<T, U>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized + 'a,
{
    type System = ProjectionSemiDiscrete<'a, Eq, T, U, Y>;

    fn discretize(self, equation: &'a Eq) -> Self::System {
        ProjectionSemiDiscrete::new(equation, self.grid, self.boundary, self.local_template)
    }
}

/// Semi-discrete ODE system produced by [`ProjectionMethod`].
#[derive(Clone, Debug)]
pub struct ProjectionSemiDiscrete<'a, Eq, T, U, Y>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: ?Sized,
{
    equation: &'a Eq,
    grid: StructuredGrid<T, 2>,
    boundary: BoundaryConditions<T, U>,
    local_template: U,
    poisson: Matrix<T>,
    marker: PhantomData<Y>,
}

impl<'a, Eq, T, U, Y> ProjectionSemiDiscrete<'a, Eq, T, U, Y>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized,
{
    pub(crate) fn new(
        equation: &'a Eq,
        grid: StructuredGrid<T, 2>,
        boundary: BoundaryConditions<T, U>,
        local_template: U,
    ) -> Self {
        assert_eq!(
            local_template.len(),
            2,
            "ProjectionMethod expects two local velocity components"
        );
        let poisson = build_poisson_matrix(&grid);
        Self {
            equation,
            grid,
            boundary,
            local_template,
            poisson,
            marker: PhantomData,
        }
    }

    /// Spatial grid used by this projection system.
    pub fn grid(&self) -> &StructuredGrid<T, 2> {
        &self.grid
    }

    fn zero_local(&self) -> U {
        self.local_template.zeros_like()
    }

    fn local_state(&self, y: &Y, node: usize) -> U {
        let mut local = self.zero_local();
        local.set_component(0, y.get_component(2 * node));
        local.set_component(1, y.get_component(2 * node + 1));
        local
    }

    fn set_local(&self, y: &mut Y, node: usize, local: &U) {
        y.set_component(2 * node, local.get_component(0));
        y.set_component(2 * node + 1, local.get_component(1));
    }

    fn is_dirichlet_node(&self, node: usize) -> bool {
        (0..2).any(|axis| {
            self.grid
                .boundary_side(node, axis)
                .and_then(|side| self.boundary.get(BoundaryFace { axis, side }))
                .is_some_and(|boundary| matches!(boundary, BoundaryCondition::Dirichlet(_)))
        })
    }

    fn directional_gradient(&self, y: &Y, node: usize, axis: usize, side: Side) -> U {
        let u = self.local_state(y, node);
        let neighbor = match side {
            Side::Lower => self.grid.neighbor(node, axis, -1),
            Side::Upper => self.grid.neighbor(node, axis, 1),
        };

        if let Some(neighbor) = neighbor {
            let other = self.local_state(y, neighbor);
            return difference(&u, &other, self.grid.dx(axis), side);
        }

        match self.boundary.get(BoundaryFace { axis, side }) {
            Some(BoundaryCondition::Neumann(gradient)) => gradient.clone(),
            Some(BoundaryCondition::Dirichlet(value)) => {
                difference(&u, value, self.grid.dx(axis), side)
            }
            None => self.zero_local(),
        }
    }

    fn raw_tendency(&self, t: T, y: &Y, dudt: &mut Y) {
        dudt.fill(T::zero());
        for node in 0..self.grid.len() {
            if self.is_dirichlet_node(node) {
                continue;
            }

            let u = self.local_state(y, node);
            let x = self.grid.point(node);
            let mut derivative = self.zero_local();
            self.equation.source(t, &x, &u, &mut derivative);

            for axis in 0..2 {
                let mut flux_lower = [self.zero_local(), self.zero_local()];
                let mut flux_upper = [self.zero_local(), self.zero_local()];
                let mut grad_lower = [self.zero_local(), self.zero_local()];
                let mut grad_upper = [self.zero_local(), self.zero_local()];
                grad_lower[axis] = self.directional_gradient(y, node, axis, Side::Lower);
                grad_upper[axis] = self.directional_gradient(y, node, axis, Side::Upper);

                self.equation.flux(t, &x, &u, &grad_lower, &mut flux_lower);
                self.equation.flux(t, &x, &u, &grad_upper, &mut flux_upper);

                for component in 0..2 {
                    derivative.set_component(
                        component,
                        derivative.get_component(component)
                            + (flux_upper[axis].get_component(component)
                                - flux_lower[axis].get_component(component))
                                / self.grid.dx(axis),
                    );
                }
            }
            self.set_local(dudt, node, &derivative);
        }
    }

    fn divergence(&self, y: &Y) -> Vec<T> {
        let mut div = vec![T::zero(); self.grid.len()];
        for node in 0..self.grid.len() {
            let [i, j] = self.grid.multi_index(node);
            let [nx, ny] = self.grid.nodes();
            let du_dx = if i == 0 {
                (y.get_component(2 * self.grid.flat_index([i + 1, j])) - y.get_component(2 * node))
                    / self.grid.dx(0)
            } else if i + 1 == nx {
                (y.get_component(2 * node) - y.get_component(2 * self.grid.flat_index([i - 1, j])))
                    / self.grid.dx(0)
            } else {
                (y.get_component(2 * self.grid.flat_index([i + 1, j]))
                    - y.get_component(2 * self.grid.flat_index([i - 1, j])))
                    / (self.grid.dx(0) + self.grid.dx(0))
            };
            let dv_dy = if j == 0 {
                (y.get_component(2 * self.grid.flat_index([i, j + 1]) + 1)
                    - y.get_component(2 * node + 1))
                    / self.grid.dx(1)
            } else if j + 1 == ny {
                (y.get_component(2 * node + 1)
                    - y.get_component(2 * self.grid.flat_index([i, j - 1]) + 1))
                    / self.grid.dx(1)
            } else {
                (y.get_component(2 * self.grid.flat_index([i, j + 1]) + 1)
                    - y.get_component(2 * self.grid.flat_index([i, j - 1]) + 1))
                    / (self.grid.dx(1) + self.grid.dx(1))
            };
            div[node] = du_dx + dv_dy;
        }
        div
    }

    fn subtract_pressure_gradient(&self, pressure: &[T], dudt: &mut Y) {
        let [nx, ny] = self.grid.nodes();
        for node in 0..self.grid.len() {
            if self.is_dirichlet_node(node) {
                dudt.set_component(2 * node, T::zero());
                dudt.set_component(2 * node + 1, T::zero());
                continue;
            }

            let [i, j] = self.grid.multi_index(node);
            let dp_dx = if i == 0 {
                (pressure[self.grid.flat_index([i + 1, j])] - pressure[node]) / self.grid.dx(0)
            } else if i + 1 == nx {
                (pressure[node] - pressure[self.grid.flat_index([i - 1, j])]) / self.grid.dx(0)
            } else {
                (pressure[self.grid.flat_index([i + 1, j])]
                    - pressure[self.grid.flat_index([i - 1, j])])
                    / (self.grid.dx(0) + self.grid.dx(0))
            };
            let dp_dy = if j == 0 {
                (pressure[self.grid.flat_index([i, j + 1])] - pressure[node]) / self.grid.dx(1)
            } else if j + 1 == ny {
                (pressure[node] - pressure[self.grid.flat_index([i, j - 1])]) / self.grid.dx(1)
            } else {
                (pressure[self.grid.flat_index([i, j + 1])]
                    - pressure[self.grid.flat_index([i, j - 1])])
                    / (self.grid.dx(1) + self.grid.dx(1))
            };

            dudt.set_component(2 * node, dudt.get_component(2 * node) - dp_dx);
            dudt.set_component(2 * node + 1, dudt.get_component(2 * node + 1) - dp_dy);
        }
    }
}

impl<Eq, T, U, Y> ODE<T, Y> for ProjectionSemiDiscrete<'_, Eq, T, U, Y>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized,
{
    fn diff(&self, t: T, y: &Y, dudt: &mut Y) {
        assert_eq!(
            y.len(),
            2 * self.grid.len(),
            "ProjectionMethod state length must match 2 * grid nodes"
        );
        assert_eq!(
            dudt.len(),
            2 * self.grid.len(),
            "ProjectionMethod derivative length must match 2 * grid nodes"
        );

        self.raw_tendency(t, y, dudt);
        let rhs = self.divergence(dudt);
        let pressure = self
            .poisson
            .lin_solve(rhs)
            .expect("projection Poisson matrix should be nonsingular");
        self.subtract_pressure_gradient(&pressure, dudt);
    }
}

fn difference<T, U>(u: &U, other: &U, dx: T, side: Side) -> U
where
    T: Real,
    U: State<T>,
{
    let mut out = u.zeros_like();
    for component in 0..u.len() {
        let value = match side {
            Side::Lower => (u.get_component(component) - other.get_component(component)) / dx,
            Side::Upper => (other.get_component(component) - u.get_component(component)) / dx,
        };
        out.set_component(component, value);
    }
    out
}

fn build_poisson_matrix<T>(grid: &StructuredGrid<T, 2>) -> Matrix<T>
where
    T: Real,
{
    let n = grid.len();
    let [nx, ny] = grid.nodes();
    let inv_dx2 = T::one() / (grid.dx(0) * grid.dx(0));
    let inv_dy2 = T::one() / (grid.dx(1) * grid.dx(1));
    let mut matrix = Matrix::full(n, n);

    for node in 0..n {
        let [i, j] = grid.multi_index(node);
        if i == 0 || j == 0 || i + 1 == nx || j + 1 == ny {
            matrix[(node, node)] = T::one();
            continue;
        }

        matrix[(node, node)] = -T::from_subset(&2.0) * (inv_dx2 + inv_dy2);
        matrix[(node, grid.flat_index([i - 1, j]))] = inv_dx2;
        matrix[(node, grid.flat_index([i + 1, j]))] = inv_dx2;
        matrix[(node, grid.flat_index([i, j - 1]))] = inv_dy2;
        matrix[(node, grid.flat_index([i, j + 1]))] = inv_dy2;
    }

    matrix
}
