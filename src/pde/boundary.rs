//! Boundary condition types for spatial PDE discretizations.

use std::marker::PhantomData;

use crate::{
    pde::{BoundaryFace, Side},
    traits::{DefaultState, Real, State},
};

/// Boundary condition for a 1D PDE.
#[derive(Clone, Debug)]
pub enum BoundaryCondition<U> {
    /// Fixed value, `u = value`.
    Dirichlet(U),
    /// Fixed spatial derivative, `du/dx = value`.
    Neumann(U),
}

/// Boundary conditions for a one-dimensional PDE domain.
#[derive(Clone, Debug)]
pub struct BoundaryConditions<T, U = DefaultState<T>>
where
    T: Real,
    U: State<T>,
{
    pub(crate) lower: Vec<Option<BoundaryCondition<U>>>,
    pub(crate) upper: Vec<Option<BoundaryCondition<U>>>,
    marker: PhantomData<T>,
}

impl<T, U> BoundaryConditions<T, U>
where
    T: Real,
    U: State<T>,
{
    /// Create boundary conditions with homogeneous Neumann boundaries.
    pub fn new() -> Self {
        Self {
            lower: Vec::new(),
            upper: Vec::new(),
            marker: PhantomData,
        }
    }

    pub(crate) fn homogeneous_neumann_like<const D: usize>(template: &U) -> Self {
        let zero = template.zeros_like();
        Self {
            lower: vec![Some(BoundaryCondition::Neumann(zero.clone())); D],
            upper: vec![Some(BoundaryCondition::Neumann(zero)); D],
            marker: PhantomData,
        }
    }

    /// Set a Dirichlet condition on one side.
    pub fn dirichlet(mut self, face: BoundaryFace, value: U) -> Self {
        self.set(face, BoundaryCondition::Dirichlet(value));
        self
    }

    /// Set a Neumann condition on one side.
    pub fn neumann(mut self, face: BoundaryFace, value: U) -> Self {
        self.set(face, BoundaryCondition::Neumann(value));
        self
    }

    /// Return the boundary condition on one side.
    pub fn get(&self, face: BoundaryFace) -> Option<&BoundaryCondition<U>> {
        self.get_face(face)
    }

    pub(crate) fn get_face(&self, face: BoundaryFace) -> Option<&BoundaryCondition<U>> {
        let boundaries = match face.side {
            Side::Lower => &self.lower,
            Side::Upper => &self.upper,
        };
        boundaries.get(face.axis).and_then(Option::as_ref)
    }

    fn set(&mut self, face: BoundaryFace, boundary: BoundaryCondition<U>) {
        let boundaries = match face.side {
            Side::Lower => &mut self.lower,
            Side::Upper => &mut self.upper,
        };
        if boundaries.len() <= face.axis {
            boundaries.resize_with(face.axis + 1, || None);
        }
        boundaries[face.axis] = Some(boundary);
    }
}

impl<T, U> Default for BoundaryConditions<T, U>
where
    T: Real,
    U: State<T>,
{
    fn default() -> Self {
        Self::new()
    }
}
