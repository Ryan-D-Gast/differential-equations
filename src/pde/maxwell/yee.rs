use std::marker::PhantomData;

use crate::{
    ode::ODE,
    pde::{BoundaryCondition, BoundaryConditions, BoundaryFace, PDE, StructuredGrid},
    traits::{Real, State},
};

use crate::pde::SpatialDiscretization;

/// Yee-grid spatial discretization backend for Maxwell's equations.
///
/// This backend implements a staggered spatial grid (FDTD/Yee scheme) for
/// resolving the curl operators in Maxwell's equations. It provides stable
/// and accurate wave propagation compared to co-located finite differences.
#[derive(Clone, Debug)]
pub struct YeeGrid<T, U, const D: usize>
where
    T: Real,
    U: State<T>,
{
    grid: StructuredGrid<T, D>,
    boundary: BoundaryConditions<T, U, D>,
    local_template: U,
}

impl<T, U> YeeGrid<T, U, 2>
where
    T: Real,
    U: State<T>,
{
    /// Create a 2D Yee grid.
    ///
    /// The local field state is assumed to be `[E_z, H_x, H_y]`.
    pub fn uniform_2d(grid: StructuredGrid<T, 2>, local_template: U) -> Self {
        Self {
            grid,
            boundary: BoundaryConditions::homogeneous_neumann_like(&local_template),
            local_template,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockMaxwell {
        c2: f64,
    }

    impl PDE<f64, Vec<f64>, 2> for MockMaxwell {
        fn flux(
            &self,
            _t: f64,
            _x: &[f64; 2],
            u: &Vec<f64>,
            _grad_u: &[Vec<f64>; 2],
            flux: &mut [Vec<f64>; 2],
        ) {
            let ez = u[0];
            let hx = u[1];
            let hy = u[2];

            flux[0][0] = self.c2 * hy;
            flux[0][1] = 0.0;
            flux[0][2] = ez;

            flux[1][0] = -self.c2 * hx;
            flux[1][1] = -ez;
            flux[1][2] = 0.0;
        }
    }

    #[test]
    fn test_yee_extract_c2() {
        let maxwell = MockMaxwell { c2: 9.0 };
        let local_template = vec![0.0; 3];
        let c2 = SemiDiscreteYee::<MockMaxwell, f64, Vec<f64>, Vec<f64>, 2>::extract_c2(
            &maxwell,
            &local_template,
        );
        assert_eq!(c2, 9.0);
    }

    #[test]
    fn test_yee_diff_interior() {
        let maxwell = MockMaxwell { c2: 1.0 };
        let grid = StructuredGrid::uniform([0.0, 0.0], [1.0, 1.0], [3, 3]);
        let local_template = vec![0.0; 3];
        let boundary = BoundaryConditions::neumann_all(vec![0.0; 3]);

        let system = SemiDiscreteYee::<_, _, _, Vec<f64>, 2>::new(
            &maxwell,
            grid.clone(),
            boundary,
            local_template,
        );

        let mut y = vec![0.0; 27]; // 3 * 3 nodes * 3 components
        let mut dudt = vec![0.0; 27];

        // Set E_z at center node (1, 1) to 1.0
        let center = grid.flat_index([1, 1]);
        y[center * 3] = 1.0;

        system.diff(0.0, &y, &mut dudt);

        // H_x at (1, 0) depends on E_z(1, 1) and E_z(1, 0).
        // d(H_x)/dt = - d(E_z)/dy = - (1.0 - 0.0) / 0.5 = -2.0
        let node_down = grid.flat_index([1, 0]);
        assert_eq!(dudt[node_down * 3 + 1], -2.0);

        // H_y at (0, 1) depends on E_z(1, 1) and E_z(0, 1).
        // d(H_y)/dt = d(E_z)/dx = (1.0 - 0.0) / 0.5 = 2.0
        let node_left = grid.flat_index([0, 1]);
        assert_eq!(dudt[node_left * 3 + 2], 2.0);
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
}

impl<'a, Eq, T, U, Y> SpatialDiscretization<'a, Eq, T, U, Y, 2> for YeeGrid<T, U, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized + 'a,
{
    type System = SemiDiscreteYee<'a, Eq, T, U, Y, 2>;

    fn discretize(self, equation: &'a Eq) -> Self::System {
        SemiDiscreteYee::new(equation, self.grid, self.boundary, self.local_template)
    }
}

/// Semi-discrete ODE system produced by the Yee grid discretization.
#[derive(Clone, Debug)]
pub struct SemiDiscreteYee<'a, Eq, T, U, Y, const D: usize>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: ?Sized,
{
    #[allow(dead_code)]
    equation: &'a Eq,
    grid: StructuredGrid<T, D>,
    boundary: BoundaryConditions<T, U, D>,
    #[allow(dead_code)]
    local_template: U,
    c2: T,
    marker: PhantomData<Y>,
}

impl<'a, Eq, T, U, Y> SemiDiscreteYee<'a, Eq, T, U, Y, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized,
{
    pub(crate) fn new(
        equation: &'a Eq,
        grid: StructuredGrid<T, 2>,
        boundary: BoundaryConditions<T, U, 2>,
        local_template: U,
    ) -> Self {
        let c2 = Self::extract_c2(equation, &local_template);

        Self {
            equation,
            grid,
            boundary,
            local_template,
            c2,
            marker: PhantomData,
        }
    }

    fn extract_c2(equation: &'a Eq, local_template: &U) -> T {
        let mut u = local_template.clone();
        u.fill(T::zero());

        // E_z = u[0], H_x = u[1], H_y = u[2]
        u.set_component(2, T::one()); // Set H_y = 1

        let grad_u = [u.zeros_like(), u.zeros_like()];
        let mut flux = [u.zeros_like(), u.zeros_like()];
        equation.flux(T::zero(), &[T::zero(), T::zero()], &u, &grad_u, &mut flux);

        // flux[0][0] should be c^2 * Hy = c^2 * 1 = c^2
        flux[0].get_component(0)
    }
}

impl<Eq, T, U, Y> ODE<T, Y> for SemiDiscreteYee<'_, Eq, T, U, Y, 2>
where
    T: Real,
    U: State<T>,
    Y: State<T>,
    Eq: PDE<T, U, 2> + ?Sized,
{
    fn diff(&self, _t: T, y: &Y, dudt: &mut Y) {
        let [nx, ny] = self.grid.nodes();
        let dx = self.grid.dx(0);
        let dy = self.grid.dx(1);
        let c2 = self.c2;

        assert_eq!(
            y.len(),
            self.grid.len() * 3,
            "YeeGrid state length must match 3 * grid nodes"
        );
        assert_eq!(
            dudt.len(),
            self.grid.len() * 3,
            "YeeGrid derivative length must match 3 * grid nodes"
        );

        dudt.fill(T::zero());

        // Field indices for the flat local field
        let ez_idx = 0;
        let hx_idx = 1;
        let hy_idx = 2;

        // 2D TM Mode Yee Grid Layout:
        // E_z is at integer grid nodes (i, j).
        // H_x is at (i, j + 1/2) -> stored at index (i, j).
        // H_y is at (i + 1/2, j) -> stored at index (i, j).

        // Update H_x: d(H_x)/dt = - d(E_z)/dy
        // H_x(i, j) uses E_z(i, j+1) and E_z(i, j)
        for i in 0..nx {
            for j in 0..ny {
                let node = self.grid.flat_index([i, j]);

                // For H_x at (i, j+1/2), derivative requires E_z at j+1 and j.
                if j + 1 < ny {
                    let node_up = self.grid.flat_index([i, j + 1]);
                    let d_ez_dy = (y.get_component(node_up * 3 + ez_idx)
                        - y.get_component(node * 3 + ez_idx))
                        / dy;
                    dudt.set_component(node * 3 + hx_idx, -d_ez_dy);
                } else {
                    dudt.set_component(node * 3 + hx_idx, T::zero());
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
                    let d_ez_dx = (y.get_component(node_right * 3 + ez_idx)
                        - y.get_component(node * 3 + ez_idx))
                        / dx;
                    dudt.set_component(node * 3 + hy_idx, d_ez_dx);
                } else {
                    dudt.set_component(node * 3 + hy_idx, T::zero());
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
                    dudt.set_component(node * 3 + ez_idx, T::zero());
                } else {
                    let d_hy_dx = if i > 0 {
                        let node_left = self.grid.flat_index([i - 1, j]);
                        (y.get_component(node * 3 + hy_idx)
                            - y.get_component(node_left * 3 + hy_idx))
                            / dx
                    } else {
                        T::zero()
                    };

                    let d_hx_dy = if j > 0 {
                        let node_down = self.grid.flat_index([i, j - 1]);
                        (y.get_component(node * 3 + hx_idx)
                            - y.get_component(node_down * 3 + hx_idx))
                            / dy
                    } else {
                        T::zero()
                    };

                    dudt.set_component(node * 3 + ez_idx, c2 * (d_hy_dx - d_hx_dy));
                }
            }
        }
    }
}
