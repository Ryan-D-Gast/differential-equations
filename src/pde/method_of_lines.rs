use crate::{
    pde::{
        BoundaryConditions, PDE, SemiDiscretePde, SpatialDiscretization, SpatialScheme,
        StructuredGrid,
    },
    traits::{DefaultState, Real, State},
};

/// Method-of-lines spatial discretization settings.
#[derive(Clone, Debug)]
pub struct MethodOfLines<T, U = DefaultState<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
{
    grid: StructuredGrid<T, D>,
    local_template: U,
    boundary: BoundaryConditions<T, U>,
    scheme: SpatialScheme,
}

impl<T> MethodOfLines<T, DefaultState<T>>
where
    T: Real,
{
    /// Create a finite-difference method-of-lines discretization for a scalar field.
    pub fn finite_difference(grid: StructuredGrid<T, 1>) -> Self {
        let local_template = T::zero();
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<1>(&local_template),
            local_template,
            scheme: SpatialScheme::FiniteDifference,
        }
    }

    /// Create a finite-volume method-of-lines discretization for a scalar field.
    pub fn finite_volume(grid: StructuredGrid<T, 1>) -> Self {
        let local_template = T::zero();
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<1>(&local_template),
            local_template,
            scheme: SpatialScheme::FiniteVolume,
        }
    }
}

impl<T, U, const D: usize> MethodOfLines<T, U, D>
where
    T: Real,
    U: State<T>,
{
    /// Create a finite-difference method-of-lines discretization on a structured grid.
    ///
    /// Dynamically sized local field types such as `Vec<T>` need this template so
    /// the discretizer can allocate per-node scratch values with the correct shape.
    pub fn finite_difference_with_field(grid: StructuredGrid<T, D>, local_template: U) -> Self {
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<D>(&local_template),
            local_template,
            scheme: SpatialScheme::FiniteDifference,
        }
    }

    /// Create a finite-volume method-of-lines discretization on a structured grid.
    pub fn finite_volume_with_field(grid: StructuredGrid<T, D>, local_template: U) -> Self {
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like::<D>(&local_template),
            local_template,
            scheme: SpatialScheme::FiniteVolume,
        }
    }

    /// Set boundary conditions.
    pub fn boundary(mut self, boundary: BoundaryConditions<T, U>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Access the spatial grid.
    pub fn grid(&self) -> &StructuredGrid<T, D> {
        &self.grid
    }

    /// Access the local field template.
    pub fn local_template(&self) -> &U {
        &self.local_template
    }

    /// Access the boundary conditions.
    pub fn boundary_conditions(&self) -> BoundaryConditions<T, U> {
        self.boundary.clone()
    }

    /// Spatial scheme used by this backend.
    pub fn scheme(&self) -> SpatialScheme {
        self.scheme
    }

    /// Convert a PDE into a semi-discrete ODE system.
    pub fn discretize<Eq, Y>(self, equation: &Eq) -> SemiDiscretePde<'_, Eq, T, U, Y, D>
    where
        Eq: PDE<T, U, D> + ?Sized,
        Y: State<T>,
    {
        SemiDiscretePde::new(
            equation,
            self.grid,
            self.local_template,
            self.boundary,
            self.scheme,
        )
    }
}

impl<'a, Eq, T, U, Y, const D: usize> SpatialDiscretization<'a, Eq, T, U, Y, D>
    for MethodOfLines<T, U, D>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, D> + ?Sized + 'a,
{
    type System = SemiDiscretePde<'a, Eq, T, U, Y, D>;

    fn discretize(self, equation: &'a Eq) -> Self::System {
        MethodOfLines::discretize(self, equation)
    }
}
