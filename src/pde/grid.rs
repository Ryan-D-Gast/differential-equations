//! Spatial grid types for PDE discretizations.

use crate::traits::Real;

/// Uniform structured grid over `D` spatial dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredGrid<T, const D: usize>
where
    T: Real,
{
    lower: [T; D],
    upper: [T; D],
    nodes: [usize; D],
    spacing: [T; D],
    len: usize,
}

impl<T, const D: usize> StructuredGrid<T, D>
where
    T: Real,
{
    /// Create a uniform structured grid with endpoint-inclusive node counts.
    pub fn uniform(lower: [T; D], upper: [T; D], nodes: [usize; D]) -> Self {
        let mut len = 1;
        let spacing = core::array::from_fn(|axis| {
            assert!(
                nodes[axis] >= 2,
                "StructuredGrid requires at least two nodes per axis"
            );
            assert!(
                lower[axis] != upper[axis],
                "StructuredGrid endpoints must be distinct"
            );
            len *= nodes[axis];
            (upper[axis] - lower[axis]) / T::from_subset(&((nodes[axis] - 1) as f64))
        });

        Self {
            lower,
            upper,
            nodes,
            spacing,
            len,
        }
    }

    /// Total number of grid nodes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the grid contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Lower coordinate bounds.
    pub fn lower(&self) -> [T; D] {
        self.lower
    }

    /// Upper coordinate bounds.
    pub fn upper(&self) -> [T; D] {
        self.upper
    }

    /// Node count along each axis.
    pub fn nodes(&self) -> [usize; D] {
        self.nodes
    }

    /// Grid spacing along each axis.
    pub fn spacing(&self) -> [T; D] {
        self.spacing
    }

    /// Grid spacing for one axis.
    pub fn dx(&self, axis: usize) -> T {
        self.spacing[axis]
    }

    /// Convert a flat node index into multi-dimensional indices.
    pub fn multi_index(&self, mut flat_index: usize) -> [usize; D] {
        assert!(flat_index < self.len, "grid index out of bounds");
        let mut index = [0; D];
        for (axis, axis_index) in index.iter_mut().enumerate() {
            *axis_index = flat_index % self.nodes[axis];
            flat_index /= self.nodes[axis];
        }
        index
    }

    /// Convert multi-dimensional indices into a flat node index.
    pub fn flat_index(&self, index: [usize; D]) -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for (axis, axis_index) in index.iter().copied().enumerate() {
            assert!(
                axis_index < self.nodes[axis],
                "grid multi-index out of bounds"
            );
            flat += axis_index * stride;
            stride *= self.nodes[axis];
        }
        flat
    }

    /// Coordinate at the given flat node index.
    pub fn point(&self, flat_index: usize) -> [T; D] {
        let index = self.multi_index(flat_index);
        core::array::from_fn(|axis| {
            self.lower[axis] + self.spacing[axis] * T::from_subset(&(index[axis] as f64))
        })
    }

    /// Iterate over grid coordinates in flat-index order.
    pub fn points(&self) -> impl Iterator<Item = [T; D]> + '_ {
        (0..self.len).map(|index| self.point(index))
    }

    pub(crate) fn neighbor(&self, flat_index: usize, axis: usize, offset: isize) -> Option<usize> {
        let mut index = self.multi_index(flat_index);
        match offset {
            -1 if index[axis] > 0 => {
                index[axis] -= 1;
                Some(self.flat_index(index))
            }
            1 if index[axis] + 1 < self.nodes[axis] => {
                index[axis] += 1;
                Some(self.flat_index(index))
            }
            _ => None,
        }
    }

    pub(crate) fn boundary_side(&self, flat_index: usize, axis: usize) -> Option<Side> {
        let index = self.multi_index(flat_index);
        if index[axis] == 0 {
            Some(Side::Lower)
        } else if index[axis] + 1 == self.nodes[axis] {
            Some(Side::Upper)
        } else {
            None
        }
    }
}

/// Side of a grid face along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Lower-coordinate side of an axis.
    Lower,
    /// Upper-coordinate side of an axis.
    Upper,
}

/// Boundary face of a structured grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryFace {
    /// Spatial axis.
    pub axis: usize,
    /// Lower or upper side of the axis.
    pub side: Side,
}

impl BoundaryFace {
    /// Create a boundary face.
    pub fn new(axis: usize, side: Side) -> Self {
        Self { axis, side }
    }

    /// Lower face for an axis.
    pub fn lower(axis: usize) -> Self {
        Self {
            axis,
            side: Side::Lower,
        }
    }

    /// Upper face for an axis.
    pub fn upper(axis: usize) -> Self {
        Self {
            axis,
            side: Side::Upper,
        }
    }
}
