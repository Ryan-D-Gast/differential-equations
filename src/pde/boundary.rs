//! Boundary condition types for spatial PDE discretizations.

use std::marker::PhantomData;

use crate::{
    pde::{BoundaryFace, Side},
    traits::{DefaultState, Real, State},
};

/// Boundary condition on one face of a PDE domain.
#[derive(Clone, Debug)]
pub enum BoundaryCondition<U> {
    /// Fixed value, `u = value`.
    Dirichlet(U),
    /// Fixed spatial derivative, `du/dx = value`.
    Neumann(U),
}

/// Complete boundary conditions for a structured PDE domain.
///
/// A `D`-dimensional structured grid has two boundary faces per axis. This type
/// stores exactly one condition for every lower and upper face, so a
/// discretization never has to infer missing public inputs. Mixed conditions are
/// allowed because many PDEs use Dirichlet conditions on some faces and Neumann
/// or flux conditions on others.
#[derive(Clone, Debug)]
pub struct BoundaryConditions<T, U = DefaultState<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
{
    pub(crate) lower: [BoundaryCondition<U>; D],
    pub(crate) upper: [BoundaryCondition<U>; D],
    marker: PhantomData<T>,
}

impl<T, U, const D: usize> BoundaryConditions<T, U, D>
where
    T: Real,
    U: State<T>,
{
    /// Create complete boundary conditions from lower and upper face arrays.
    pub fn new(lower: [BoundaryCondition<U>; D], upper: [BoundaryCondition<U>; D]) -> Self {
        Self {
            lower,
            upper,
            marker: PhantomData,
        }
    }

    pub(crate) fn homogeneous_neumann_like(template: &U) -> Self {
        let zero = template.zeros_like();
        Self {
            lower: core::array::from_fn(|_| BoundaryCondition::Neumann(zero.clone())),
            upper: core::array::from_fn(|_| BoundaryCondition::Neumann(zero.clone())),
            marker: PhantomData,
        }
    }

    /// Create boundary conditions where every face is Dirichlet.
    pub fn dirichlet_all(value: U) -> Self {
        Self {
            lower: core::array::from_fn(|_| BoundaryCondition::Dirichlet(value.clone())),
            upper: core::array::from_fn(|_| BoundaryCondition::Dirichlet(value.clone())),
            marker: PhantomData,
        }
    }

    /// Create boundary conditions where every face is Neumann.
    pub fn neumann_all(value: U) -> Self {
        Self {
            lower: core::array::from_fn(|_| BoundaryCondition::Neumann(value.clone())),
            upper: core::array::from_fn(|_| BoundaryCondition::Neumann(value.clone())),
            marker: PhantomData,
        }
    }

    /// Start an explicit builder for mixed boundary conditions.
    pub fn builder() -> BoundaryConditionsBuilder<T, U, D> {
        BoundaryConditionsBuilder::new()
    }

    /// Return the boundary condition on one side.
    pub fn get(&self, face: BoundaryFace) -> &BoundaryCondition<U> {
        self.get_face(face)
    }

    pub(crate) fn get_face(&self, face: BoundaryFace) -> &BoundaryCondition<U> {
        assert!(face.axis < D, "boundary face axis out of bounds");
        match face.side {
            Side::Lower => &self.lower[face.axis],
            Side::Upper => &self.upper[face.axis],
        }
    }
}

/// Builder error for incomplete boundary-condition specifications.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryConditionsBuilderError {
    /// Missing boundary face.
    pub face: BoundaryFace,
}

/// Builder for complete mixed boundary conditions.
#[derive(Clone, Debug)]
pub struct BoundaryConditionsBuilder<T, U = DefaultState<T>, const D: usize = 1>
where
    T: Real,
    U: State<T>,
{
    lower: [Option<BoundaryCondition<U>>; D],
    upper: [Option<BoundaryCondition<U>>; D],
    marker: PhantomData<T>,
}

impl<T, U, const D: usize> BoundaryConditionsBuilder<T, U, D>
where
    T: Real,
    U: State<T>,
{
    /// Create an empty boundary-condition builder.
    pub fn new() -> Self {
        Self {
            lower: core::array::from_fn(|_| None),
            upper: core::array::from_fn(|_| None),
            marker: PhantomData,
        }
    }

    /// Set a Dirichlet condition on one face.
    pub fn dirichlet(mut self, face: BoundaryFace, value: U) -> Self {
        self.set(face, BoundaryCondition::Dirichlet(value));
        self
    }

    /// Set a Neumann condition on one face.
    pub fn neumann(mut self, face: BoundaryFace, value: U) -> Self {
        self.set(face, BoundaryCondition::Neumann(value));
        self
    }

    /// Build complete boundary conditions.
    pub fn build(self) -> Result<BoundaryConditions<T, U, D>, BoundaryConditionsBuilderError> {
        let lower = Self::complete_side(self.lower, Side::Lower)?;
        let upper = Self::complete_side(self.upper, Side::Upper)?;
        Ok(BoundaryConditions::new(lower, upper))
    }

    fn set(&mut self, face: BoundaryFace, boundary: BoundaryCondition<U>) {
        assert!(face.axis < D, "boundary face axis out of bounds");
        let boundaries = match face.side {
            Side::Lower => &mut self.lower,
            Side::Upper => &mut self.upper,
        };
        boundaries[face.axis] = Some(boundary);
    }

    fn complete_side(
        side: [Option<BoundaryCondition<U>>; D],
        boundary_side: Side,
    ) -> Result<[BoundaryCondition<U>; D], BoundaryConditionsBuilderError> {
        for (axis, boundary) in side.iter().enumerate() {
            if boundary.is_none() {
                return Err(BoundaryConditionsBuilderError {
                    face: BoundaryFace {
                        axis,
                        side: boundary_side,
                    },
                });
            }
        }

        Ok(side.map(|boundary| {
            boundary
                .expect("all boundary faces were checked before constructing BoundaryConditions")
        }))
    }
}

impl<T, U, const D: usize> Default for BoundaryConditionsBuilder<T, U, D>
where
    T: Real,
    U: State<T>,
{
    fn default() -> Self {
        Self::new()
    }
}
