use crate::{
    ode::ODE,
    pde::{
        BoundaryCondition, BoundaryConditions, BoundaryFace, PDE, Side, SpatialDiscretization,
        StructuredGrid,
    },
    traits::{DefaultState, Real, State},
};

use super::{Limiter, NumericalFlux, Reconstruction};

/// Finite Volume spatial discretization backend.
#[derive(Clone, Debug)]
pub struct FiniteVolume<T, U = DefaultState<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
{
    grid: StructuredGrid<T, D>,
    local_template: U,
    boundary: BoundaryConditions<T, U>,
    reconstruction: Reconstruction,
    flux: NumericalFlux,
    limiter: Limiter,
}

impl<T> FiniteVolume<T, DefaultState<T>>
where
    T: Real,
{
    /// Create a finite volume discretization on a structured grid for a scalar field.
    pub fn structured(grid: StructuredGrid<T, 1>) -> Self {
        let local_template = T::zero();
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<1>(&local_template),
            local_template,
            reconstruction: Reconstruction::default(),
            flux: NumericalFlux::default(),
            limiter: Limiter::default(),
        }
    }
}

impl<T, U, const D: usize> FiniteVolume<T, U, D>
where
    T: Real,
    U: State<T>,
{
    /// Create a finite volume discretization on a structured grid.
    pub fn structured_with_field(grid: StructuredGrid<T, D>, local_template: U) -> Self {
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<D>(&local_template),
            local_template,
            reconstruction: Reconstruction::default(),
            flux: NumericalFlux::default(),
            limiter: Limiter::default(),
        }
    }

    /// Set boundary conditions.
    pub fn boundary(mut self, boundary: BoundaryConditions<T, U>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set reconstruction method.
    pub fn reconstruction(mut self, reconstruction: Reconstruction) -> Self {
        self.reconstruction = reconstruction;
        self
    }

    /// Set numerical flux.
    pub fn flux(mut self, flux: NumericalFlux) -> Self {
        self.flux = flux;
        self
    }

    /// Set limiter.
    pub fn limiter(mut self, limiter: Limiter) -> Self {
        self.limiter = limiter;
        self
    }

    /// Convert a PDE into a semi-discrete ODE system.
    pub fn discretize<Eq, Y>(self, equation: &Eq) -> FiniteVolumeSemiDiscrete<'_, Eq, T, U, Y, D>
    where
        Eq: PDE<T, U, D> + ?Sized,
        Y: State<T>,
    {
        FiniteVolumeSemiDiscrete {
            equation,
            grid: self.grid,
            local_template: self.local_template,
            boundary: self.boundary,
            reconstruction: self.reconstruction,
            flux: self.flux,
            limiter: self.limiter,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Eq, T, U, Y, const D: usize> SpatialDiscretization<'a, Eq, T, U, Y, D>
    for FiniteVolume<T, U, D>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, D> + ?Sized + 'a,
{
    type System = FiniteVolumeSemiDiscrete<'a, Eq, T, U, Y, D>;

    fn discretize(self, equation: &'a Eq) -> Self::System {
        FiniteVolume::discretize(self, equation)
    }
}

/// Semi-discrete system for finite volume.
#[derive(Clone, Debug)]
pub struct FiniteVolumeSemiDiscrete<'a, Eq, T, U = T, Y = Vec<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: ?Sized,
{
    pub(crate) equation: &'a Eq,
    pub(crate) grid: StructuredGrid<T, D>,
    pub(crate) local_template: U,
    pub(crate) boundary: BoundaryConditions<T, U>,
    pub(crate) reconstruction: Reconstruction,
    pub(crate) flux: NumericalFlux,
    pub(crate) limiter: Limiter,
    pub(crate) marker: std::marker::PhantomData<Y>,
}

impl<'a, Eq, T, U, Y, const D: usize> FiniteVolumeSemiDiscrete<'a, Eq, T, U, Y, D>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, D> + ?Sized,
{
    fn local_len(&self) -> usize {
        self.local_template.len()
    }

    fn zero_local(&self) -> U {
        self.local_template.zeros_like()
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

    fn face_boundary(&self, axis: usize, side: Side) -> Option<&BoundaryCondition<U>> {
        self.boundary.get_face(BoundaryFace { axis, side })
    }

    fn ghost_state(&self, y: &Y, node: usize, axis: usize, side: Side, distance: isize) -> U {
        let neighbor = match side {
            Side::Lower => self.grid.neighbor(node, axis, -distance),
            Side::Upper => self.grid.neighbor(node, axis, distance),
        };

        if let Some(neighbor) = neighbor {
            self.local_state(y, neighbor)
        } else {
            // Extrapolate or apply boundary condition.
            let u = self.local_state(y, node);
            let bnd_side = self.grid.boundary_side(node, axis).unwrap_or(side);
            match self.face_boundary(axis, bnd_side) {
                Some(BoundaryCondition::Dirichlet(val)) => {
                    // Simple reflection/extrapolation for Dirichlet to keep symmetry or use exact value.
                    // For distance > 1, this is an approximation.
                    let mut ext = val.clone();
                    for i in 0..self.local_len() {
                        ext.set_component(
                            i,
                            T::from_subset(&2.0) * val.get_component(i) - u.get_component(i),
                        );
                    }
                    ext
                }
                Some(BoundaryCondition::Neumann(grad)) => {
                    let mut ext = u.clone();
                    let dx = self.grid.dx(axis) * T::from_subset(&(distance as f64));
                    for i in 0..self.local_len() {
                        let sign = match bnd_side {
                            Side::Lower => -T::one(),
                            Side::Upper => T::one(),
                        };
                        ext.set_component(
                            i,
                            u.get_component(i) + sign * grad.get_component(i) * dx,
                        );
                    }
                    ext
                }
                None => u,
            }
        }
    }
}

impl<'a, Eq, T, U, Y, const D: usize> ODE<T, Y> for FiniteVolumeSemiDiscrete<'a, Eq, T, U, Y, D>
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

        // Precompute all source terms and initialize dudt.
        for node in 0..self.grid.len() {
            let u = self.local_state(y, node);
            let x = self.grid.point(node);
            let mut derivative = self.zero_local();
            self.equation.source(t, &x, &u, &mut derivative);
            self.set_local_derivative(dudt, node, &derivative);
        }

        // Finite volume represents conservation laws as `u_t + div(F) = source`.
        // Each face flux is added to one adjacent cell and subtracted from the other.

        for axis in 0..D {
            // Compute fluxes at each face.
            // Face j+1/2 is between node j and neighbor(j, axis, 1).
            for node in 0..self.grid.len() {
                // We only compute the face flux for the upper face of each node.
                // Lower faces are covered by the neighbor's upper face, except at the very lower boundary.

                let neighbor_right = self.grid.neighbor(node, axis, 1);

                let (u_l, u_r) = if let Some(nr) = neighbor_right {
                    let u_ll = self.ghost_state(y, node, axis, Side::Lower, 1);
                    let u_l = self.local_state(y, node);
                    let u_r = self.local_state(y, nr);
                    let u_rr = self.ghost_state(y, nr, axis, Side::Upper, 1);

                    self.reconstruction
                        .reconstruct::<T, U>(&u_ll, &u_l, &u_r, &u_rr, &self.limiter)
                } else {
                    // Right boundary face
                    let u_ll = self.ghost_state(y, node, axis, Side::Lower, 1);
                    let u_l = self.local_state(y, node);
                    let u_r = self.ghost_state(y, node, axis, Side::Upper, 1);
                    let u_rr = self.ghost_state(y, node, axis, Side::Upper, 2);

                    self.reconstruction
                        .reconstruct::<T, U>(&u_ll, &u_l, &u_r, &u_rr, &self.limiter)
                };

                let mut x_face = self.grid.point(node);
                x_face[axis] += self.grid.dx(axis) * T::from_subset(&0.5);

                let flux_upper = self
                    .flux
                    .compute(self.equation, t, &x_face, &u_l, &u_r, axis);

                let mut current_deriv = self.zero_local();
                for i in 0..local_len {
                    current_deriv.set_component(
                        i,
                        dudt.get_component(node * local_len + i)
                            - flux_upper.get_component(i) / self.grid.dx(axis),
                    );
                }
                self.set_local_derivative(dudt, node, &current_deriv);

                if let Some(nr) = neighbor_right {
                    let mut neighbor_deriv = self.zero_local();
                    for i in 0..local_len {
                        neighbor_deriv.set_component(
                            i,
                            dudt.get_component(nr * local_len + i)
                                + flux_upper.get_component(i) / self.grid.dx(axis),
                        );
                    }
                    self.set_local_derivative(dudt, nr, &neighbor_deriv);
                }

                // If node is at the lower boundary, we also need to compute the lower boundary face flux explicitly.
                if self.grid.neighbor(node, axis, -1).is_none() {
                    let u_ll = self.ghost_state(y, node, axis, Side::Lower, 2);
                    let u_l = self.ghost_state(y, node, axis, Side::Lower, 1);
                    let u_r = self.local_state(y, node);
                    let u_rr = self.ghost_state(y, node, axis, Side::Upper, 1);

                    let (u_l_face, u_r_face) = self.reconstruction.reconstruct::<T, U>(
                        &u_ll,
                        &u_l,
                        &u_r,
                        &u_rr,
                        &self.limiter,
                    );

                    let mut x_face_lower = self.grid.point(node);
                    x_face_lower[axis] -= self.grid.dx(axis) * T::from_subset(&0.5);

                    let flux_lower = self.flux.compute(
                        self.equation,
                        t,
                        &x_face_lower,
                        &u_l_face,
                        &u_r_face,
                        axis,
                    );

                    let mut current_deriv = self.zero_local();
                    for i in 0..local_len {
                        current_deriv.set_component(
                            i,
                            dudt.get_component(node * local_len + i)
                                + flux_lower.get_component(i) / self.grid.dx(axis),
                        );
                    }
                    self.set_local_derivative(dudt, node, &current_deriv);
                }
            }
        }

        // Zero out derivative for Dirichlet nodes
        for node in 0..self.grid.len() {
            let is_dirichlet = (0..D).any(|axis| {
                self.grid
                    .boundary_side(node, axis)
                    .and_then(|side| self.face_boundary(axis, side))
                    .is_some_and(|boundary| matches!(boundary, BoundaryCondition::Dirichlet(_)))
            });
            if is_dirichlet {
                self.set_local_derivative(dudt, node, &self.zero_local());
            }
        }
    }
}
