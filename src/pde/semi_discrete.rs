//! Semi-discrete ODE systems produced by PDE spatial discretization.

use std::marker::PhantomData;

use crate::{
    ode::ODE,
    pde::{BoundaryCondition, BoundaryConditions, BoundaryFace, PDE, Side, StructuredGrid},
    traits::{Real, State},
};

/// Spatial method used by the method-of-lines backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialScheme {
    /// Central finite-difference style flux divergence.
    FiniteDifference,
    /// Cell-centered finite-volume style flux balance.
    FiniteVolume,
}

/// Semi-discrete ODE system produced by method of lines.
#[derive(Clone, Debug)]
pub struct SemiDiscretePde<'a, Eq, T, U = T, Y = Vec<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: ?Sized,
{
    equation: &'a Eq,
    grid: StructuredGrid<T, D>,
    local_template: U,
    boundary: BoundaryConditions<T, U, D>,
    scheme: SpatialScheme,
    marker: PhantomData<Y>,
}

impl<'a, Eq, T, U, Y, const D: usize> SemiDiscretePde<'a, Eq, T, U, Y, D>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, D> + ?Sized,
{
    pub(crate) fn new(
        equation: &'a Eq,
        grid: StructuredGrid<T, D>,
        local_template: U,
        boundary: BoundaryConditions<T, U, D>,
        scheme: SpatialScheme,
    ) -> Self {
        Self {
            equation,
            grid,
            local_template,
            boundary,
            scheme,
            marker: PhantomData,
        }
    }

    /// Spatial grid used by this semi-discrete system.
    pub fn grid(&self) -> &StructuredGrid<T, D> {
        &self.grid
    }

    /// Boundary conditions used by this semi-discrete system.
    pub fn boundary_conditions(&self) -> BoundaryConditions<T, U, D> {
        self.boundary.clone()
    }

    /// Number of scalar components in each local field value.
    pub fn local_len(&self) -> usize {
        self.local_template.len()
    }

    /// Spatial scheme used by this semi-discrete system.
    pub fn scheme(&self) -> SpatialScheme {
        self.scheme
    }

    fn zero_local(&self) -> U {
        self.local_template.zeros_like()
    }

    fn zero_gradient(&self) -> [U; D] {
        core::array::from_fn(|_| self.zero_local())
    }

    fn local_state(&self, y: &Y, node: usize) -> U {
        let local_len = self.local_len();
        let mut local = self.zero_local();
        for component in 0..local_len {
            local.set_component(component, y.get_component(node * local_len + component));
        }
        local
    }

    fn set_local_derivative(&self, dudt: &mut Y, node: usize, derivative: &U) {
        let local_len = self.local_len();
        for component in 0..local_len {
            dudt.set_component(
                node * local_len + component,
                derivative.get_component(component),
            );
        }
    }

    fn difference(&self, right: &U, left: &U, scale: T) -> U {
        let mut out = self.zero_local();
        for component in 0..self.local_len() {
            out.set_component(
                component,
                (right.get_component(component) - left.get_component(component)) / scale,
            );
        }
        out
    }

    fn add_scaled_difference(&self, out: &mut U, right: &U, left: &U, scale: T) {
        for component in 0..self.local_len() {
            out.set_component(
                component,
                out.get_component(component)
                    + (right.get_component(component) - left.get_component(component)) / scale,
            );
        }
    }

    fn face_boundary(&self, axis: usize, side: Side) -> &BoundaryCondition<U> {
        self.boundary.get_face(BoundaryFace { axis, side })
    }

    fn is_dirichlet_node(&self, node: usize) -> bool {
        (0..D).any(|axis| {
            self.grid.boundary_side(node, axis).is_some_and(|side| {
                matches!(
                    self.face_boundary(axis, side),
                    BoundaryCondition::Dirichlet(_)
                )
            })
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
            return match side {
                Side::Lower => self.difference(&u, &other, self.grid.dx(axis)),
                Side::Upper => self.difference(&other, &u, self.grid.dx(axis)),
            };
        }

        match self.face_boundary(axis, side) {
            BoundaryCondition::Neumann(gradient) => gradient.clone(),
            BoundaryCondition::Dirichlet(value) => match side {
                Side::Lower => self.difference(&u, value, self.grid.dx(axis)),
                Side::Upper => self.difference(value, &u, self.grid.dx(axis)),
            },
        }
    }

    fn directional_flux(&self, t: T, y: &Y, node: usize, axis: usize, side: Side) -> U {
        match self.scheme {
            SpatialScheme::FiniteDifference => self.finite_difference_flux(t, y, node, axis, side),
            SpatialScheme::FiniteVolume => self.finite_volume_flux(t, y, node, axis, side),
        }
    }

    fn finite_difference_flux(&self, t: T, y: &Y, node: usize, axis: usize, side: Side) -> U {
        let u = self.local_state(y, node);
        let x = self.grid.point(node);
        let mut grad_u = self.zero_gradient();
        grad_u[axis] = self.directional_gradient(y, node, axis, side);

        let mut flux = core::array::from_fn(|_| self.zero_local());
        self.equation.flux(t, &x, &u, &grad_u, &mut flux);
        flux[axis].clone()
    }

    fn finite_volume_flux(&self, t: T, y: &Y, node: usize, axis: usize, side: Side) -> U {
        let u = self.local_state(y, node);
        let x_node = self.grid.point(node);
        let neighbor = match side {
            Side::Lower => self.grid.neighbor(node, axis, -1),
            Side::Upper => self.grid.neighbor(node, axis, 1),
        };

        let (u_face, grad_axis, x_face) = if let Some(neighbor) = neighbor {
            let other = self.local_state(y, neighbor);
            let mut u_face = self.zero_local();
            for component in 0..self.local_len() {
                u_face.set_component(
                    component,
                    (u.get_component(component) + other.get_component(component))
                        / T::from_subset(&2.0),
                );
            }
            let grad_axis = match side {
                Side::Lower => self.difference(&u, &other, self.grid.dx(axis)),
                Side::Upper => self.difference(&other, &u, self.grid.dx(axis)),
            };
            let mut x_face = x_node;
            let offset = self.grid.dx(axis) / T::from_subset(&2.0);
            match side {
                Side::Lower => x_face[axis] -= offset,
                Side::Upper => x_face[axis] += offset,
            }
            (u_face, grad_axis, x_face)
        } else {
            let side = self.grid.boundary_side(node, axis).unwrap_or(side);
            match self.face_boundary(axis, side) {
                BoundaryCondition::Dirichlet(value) => {
                    let grad_axis = match side {
                        Side::Lower => self.difference(&u, value, self.grid.dx(axis)),
                        Side::Upper => self.difference(value, &u, self.grid.dx(axis)),
                    };
                    (value.clone(), grad_axis, x_node)
                }
                BoundaryCondition::Neumann(gradient) => (u, gradient.clone(), x_node),
            }
        };

        let mut grad_u = self.zero_gradient();
        grad_u[axis] = grad_axis;
        let mut flux = core::array::from_fn(|_| self.zero_local());
        self.equation.flux(t, &x_face, &u_face, &grad_u, &mut flux);
        flux[axis].clone()
    }
}

impl<Eq, T, U, Y, const D: usize> ODE<T, Y> for SemiDiscretePde<'_, Eq, T, U, Y, D>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, D> + ?Sized,
{
    fn diff(&self, t: T, y: &Y, dudt: &mut Y) {
        let local_len = self.local_len();
        assert_eq!(
            y.len(),
            self.grid.len() * local_len,
            "PDE state length must match the spatial grid"
        );
        assert_eq!(
            dudt.len(),
            self.grid.len() * local_len,
            "PDE derivative length must match the spatial grid"
        );

        for node in 0..self.grid.len() {
            if self.is_dirichlet_node(node) {
                self.set_local_derivative(dudt, node, &self.zero_local());
                continue;
            }

            let u = self.local_state(y, node);
            let x = self.grid.point(node);
            let mut derivative = self.zero_local();
            self.equation.source(t, &x, &u, &mut derivative);

            for axis in 0..D {
                let flux_lower = self.directional_flux(t, y, node, axis, Side::Lower);
                let flux_upper = self.directional_flux(t, y, node, axis, Side::Upper);
                self.add_scaled_difference(
                    &mut derivative,
                    &flux_upper,
                    &flux_lower,
                    self.grid.dx(axis),
                );
            }

            self.set_local_derivative(dudt, node, &derivative);
        }
    }
}
