use std::marker::PhantomData;

use crate::{
    ode::ODE,
    pde::{BoundaryCondition, BoundaryConditions, BoundaryFace, StructuredGrid},
    traits::{Real, State},
};

use crate::pde::SpatialDiscretization;

/// Component layout for a two-dimensional Yee staggered-grid update.
///
/// The default layout is `[electric_z, magnetic_x, magnetic_y] = [0, 1, 2]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YeeLayout {
    /// Component updated by the staggered curl of the two magnetic components.
    pub electric_z: usize,
    /// Component updated from the y-derivative of `electric_z`.
    pub magnetic_x: usize,
    /// Component updated from the x-derivative of `electric_z`.
    pub magnetic_y: usize,
}

impl Default for YeeLayout {
    fn default() -> Self {
        Self {
            electric_z: 0,
            magnetic_x: 1,
            magnetic_y: 2,
        }
    }
}

/// Two-dimensional Yee staggered-grid spatial discretization backend.
///
/// This backend implements the standard 2D TM-style staggered curl update for
/// three selected local components. It is configured by component indices and
/// wave speed, so it is not tied to a particular equation type.
#[derive(Clone, Debug)]
pub struct YeeGrid<T, U, const D: usize>
where
    T: Real,
    U: State<T>,
{
    grid: StructuredGrid<T, D>,
    boundary: BoundaryConditions<T, U, D>,
    local_template: U,
    layout: YeeLayout,
    wave_speed_squared: T,
}

impl<T, U> YeeGrid<T, U, 2>
where
    T: Real,
    U: State<T>,
{
    /// Create a 2D Yee grid.
    ///
    /// The default component layout is `[electric_z, magnetic_x, magnetic_y] = [0, 1, 2]`
    /// and the default wave speed is one.
    pub fn uniform_2d(grid: StructuredGrid<T, 2>, local_template: U) -> Self {
        assert!(
            local_template.len() >= 3,
            "YeeGrid requires at least three local components"
        );
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like(&local_template),
            local_template,
            layout: YeeLayout::default(),
            wave_speed_squared: T::one(),
        }
    }
}

impl<T, U, const D: usize> YeeGrid<T, U, D>
where
    T: Real,
    U: State<T>,
{
    /// Set boundary conditions.
    pub fn boundary(mut self, boundary: BoundaryConditions<T, U, D>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the local component layout used by the Yee update.
    pub fn layout(mut self, layout: YeeLayout) -> Self {
        assert!(
            layout.electric_z < self.local_template.len()
                && layout.magnetic_x < self.local_template.len()
                && layout.magnetic_y < self.local_template.len(),
            "YeeGrid layout component index out of bounds"
        );
        assert!(
            layout.electric_z != layout.magnetic_x
                && layout.electric_z != layout.magnetic_y
                && layout.magnetic_x != layout.magnetic_y,
            "YeeGrid layout components must be distinct"
        );
        self.layout = layout;
        self
    }

    /// Set the squared wave speed used in the electric-field update.
    pub fn wave_speed_squared(mut self, wave_speed_squared: T) -> Self {
        self.wave_speed_squared = wave_speed_squared;
        self
    }

    /// Set the wave speed used in the electric-field update.
    pub fn wave_speed(mut self, wave_speed: T) -> Self {
        self.wave_speed_squared = wave_speed * wave_speed;
        self
    }
}

impl<Eq, T, U, Y> SpatialDiscretization<Eq, T, U, Y, 2> for YeeGrid<T, U, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
{
    type System = SemiDiscreteYee<Eq, T, U, Y, 2>;

    fn discretize(self, equation: Eq) -> Self::System {
        SemiDiscreteYee::new(
            equation,
            self.grid,
            self.boundary,
            self.local_template,
            self.layout,
            self.wave_speed_squared,
        )
    }
}

/// Semi-discrete ODE system produced by the Yee grid discretization.
#[derive(Clone, Debug)]
pub struct SemiDiscreteYee<Eq, T, U, Y, const D: usize>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
{
    #[allow(dead_code)]
    equation: Eq,
    grid: StructuredGrid<T, D>,
    boundary: BoundaryConditions<T, U, D>,
    #[allow(dead_code)]
    local_template: U,
    layout: YeeLayout,
    wave_speed_squared: T,
    marker: PhantomData<Y>,
}

impl<Eq, T, U, Y> SemiDiscreteYee<Eq, T, U, Y, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
{
    pub(crate) fn new(
        equation: Eq,
        grid: StructuredGrid<T, 2>,
        boundary: BoundaryConditions<T, U, 2>,
        local_template: U,
        layout: YeeLayout,
        wave_speed_squared: T,
    ) -> Self {
        Self {
            equation,
            grid,
            boundary,
            local_template,
            layout,
            wave_speed_squared,
            marker: PhantomData,
        }
    }
}

impl<Eq, T, U, Y> ODE<T, Y> for SemiDiscreteYee<Eq, T, U, Y, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
{
    fn diff(&self, _t: T, y: &Y, dudt: &mut Y) {
        let [nx, ny] = self.grid.nodes();
        let dx = self.grid.dx(0);
        let dy = self.grid.dx(1);
        let c2 = self.wave_speed_squared;
        let layout = self.layout;
        let local_len = self.local_template.len();

        assert_eq!(
            y.len(),
            self.grid.len() * local_len,
            "YeeGrid state length must match local components * grid nodes"
        );
        assert_eq!(
            dudt.len(),
            self.grid.len() * local_len,
            "YeeGrid derivative length must match local components * grid nodes"
        );

        dudt.fill(T::zero());

        for i in 0..nx {
            for j in 0..ny {
                let node = self.grid.flat_index([i, j]);

                // For H_x at (i, j+1/2), derivative requires E_z at j+1 and j.
                if j + 1 < ny {
                    let node_up = self.grid.flat_index([i, j + 1]);
                    let d_ez_dy = (y.get_component(node_up * local_len + layout.electric_z)
                        - y.get_component(node * local_len + layout.electric_z))
                        / dy;
                    dudt.set_component(node * local_len + layout.magnetic_x, -d_ez_dy);
                } else {
                    dudt.set_component(node * local_len + layout.magnetic_x, T::zero());
                }
            }
        }

        // Update H_y: d(H_y)/dt = d(E_z)/dx
        // H_y(i, j) uses E_z(i+1, j) and E_z(i, j)
        for i in 0..nx {
            for j in 0..ny {
                let node = self.grid.flat_index([i, j]);

                if i + 1 < nx {
                    let node_right = self.grid.flat_index([i + 1, j]);
                    let d_ez_dx = (y.get_component(node_right * local_len + layout.electric_z)
                        - y.get_component(node * local_len + layout.electric_z))
                        / dx;
                    dudt.set_component(node * local_len + layout.magnetic_y, d_ez_dx);
                } else {
                    dudt.set_component(node * local_len + layout.magnetic_y, T::zero());
                }
            }
        }

        // Update E_z: d(E_z)/dt = c^2 * (d(H_y)/dx - d(H_x)/dy)
        // E_z(i, j) uses H_y(i, j) and H_y(i-1, j), and H_x(i, j) and H_x(i, j-1)
        for i in 0..nx {
            for j in 0..ny {
                let node = self.grid.flat_index([i, j]);

                // Boundaries for E_z
                let mut is_dirichlet = false;
                for axis in 0..2 {
                    if let Some(side) = self.grid.boundary_side(node, axis) {
                        let face = BoundaryFace { axis, side };
                        if matches!(self.boundary.get(face), BoundaryCondition::Dirichlet(_)) {
                            is_dirichlet = true;
                            break;
                        }
                    }
                }

                if is_dirichlet {
                    dudt.set_component(node * local_len + layout.electric_z, T::zero());
                } else {
                    let d_hy_dx = if i > 0 {
                        let node_left = self.grid.flat_index([i - 1, j]);
                        (y.get_component(node * local_len + layout.magnetic_y)
                            - y.get_component(node_left * local_len + layout.magnetic_y))
                            / dx
                    } else {
                        T::zero()
                    };

                    let d_hx_dy = if j > 0 {
                        let node_down = self.grid.flat_index([i, j - 1]);
                        (y.get_component(node * local_len + layout.magnetic_x)
                            - y.get_component(node_down * local_len + layout.magnetic_x))
                            / dy
                    } else {
                        T::zero()
                    };

                    dudt.set_component(
                        node * local_len + layout.electric_z,
                        c2 * (d_hy_dx - d_hx_dy),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yee_configuration() {
        let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [3, 3]);
        let system: SemiDiscreteYee<_, _, _, Vec<f64>, 2> = YeeGrid::uniform_2d(grid, vec![0.0; 4])
            .layout(YeeLayout {
                electric_z: 2,
                magnetic_x: 0,
                magnetic_y: 1,
            })
            .wave_speed_squared(9.0)
            .discretize(&());

        assert_eq!(system.layout.electric_z, 2);
        assert_eq!(system.wave_speed_squared, 9.0);
    }

    #[test]
    fn test_yee_diff_interior() {
        let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [3, 3]);
        let local_template = vec![0.0; 3];
        let boundary = BoundaryConditions::neumann_all(vec![0.0; 3]);

        let system = SemiDiscreteYee::<_, _, _, Vec<f64>, 2>::new(
            &(),
            grid.clone(),
            boundary,
            local_template,
            YeeLayout::default(),
            1.0,
        );

        let mut y = vec![0.0; 27];
        let mut dudt = vec![0.0; 27];

        let center = grid.flat_index([1, 1]);
        y[center * 3] = 1.0;

        system.diff(0.0, &y, &mut dudt);

        let node_down = grid.flat_index([1, 0]);
        assert_eq!(dudt[node_down * 3 + 1], -2.0);

        let node_left = grid.flat_index([0, 1]);
        assert_eq!(dudt[node_left * 3 + 2], 2.0);
    }

    #[test]
    fn test_yee_wave_speed_scales_electric_update() {
        let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [3, 3]);
        let local_template = vec![0.0; 3];
        let boundary = BoundaryConditions::neumann_all(vec![0.0; 3]);

        let system = SemiDiscreteYee::<_, _, _, Vec<f64>, 2>::new(
            &(),
            grid.clone(),
            boundary,
            local_template,
            YeeLayout::default(),
            9.0,
        );

        let mut y = vec![0.0; 27];
        let mut dudt = vec![0.0; 27];

        let center = grid.flat_index([1, 1]);
        y[center * 3 + 2] = 1.0;

        system.diff(0.0, &y, &mut dudt);

        assert_eq!(dudt[center * 3], 18.0);
    }
}
