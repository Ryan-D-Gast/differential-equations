//! Diagonally Implicit Runge-Kutta (DIRK) tableau

use crate::{tableau::ButcherTableau, traits::Real};

impl<T: Real> ButcherTableau<T, 2> {
    /// SDIRK-2-1: 2-stage, 2nd order SDIRK method with 1st order embedding
    ///
    /// # Overview
    /// This provides a 2-stage, singly diagonally implicit Runge-Kutta method with:
    /// - Primary order: 2
    /// - Embedded order: 1 (for error estimation)
    /// - Number of stages: 2
    /// - A-stable and B-stable
    ///
    /// # Notes
    /// - This is a simple SDIRK method where all diagonal entries are equal (γ = 1)
    /// - Good for basic adaptive stepping with stiff problems
    /// - The embedded method provides basic error estimation for step size control
    /// - Particularly useful as a starting method for more complex stiff systems
    ///
    /// # Butcher Tableau
    /// ```text
    /// 1    | 1    0
    /// 0    | -1   1
    /// -----|--------
    ///      | 1/2  1/2  (2nd order)
    ///      | 1    0    (1st order embedding)
    /// ```
    ///
    /// # References
    /// - Hairer, E., Wanner, G. (1996). "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems"
    ///
    pub fn sdirk21() -> Self {
        let c = [1.0, 0.0];
        let a = [[1.0, 0.0], [-1.0, 1.0]];
        let b = [0.5, 0.5];
        let bh = [1.0, 0.0]; // 1st order embedding

        let c = c.map(|x| T::from_f64(x).unwrap());
        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let bh = Some(bh.map(|x| T::from_f64(x).unwrap()));

        ButcherTableau {
            c,
            a,
            b,
            bh,
            bi: None,
            er: None,
        }
    }
}

impl<T: Real> ButcherTableau<T, 3> {
    /// ESDIRK-3-3: 3-stage, 3rd order ESDIRK method
    ///
    /// # Overview
    /// This provides a 3-stage, explicit singly diagonally implicit Runge-Kutta method with:
    /// - Primary order: 3
    /// - Number of stages: 3
    /// - A-stable
    ///
    /// # Notes
    /// - This method pairs with SSPRK(3,3)-Shu-Osher-ERK to make a 3rd order IMEX method
    /// - Has an explicit first stage (ESDIRK property) making it computationally efficient
    /// - The first stage being explicit reduces the computational cost per step
    /// - Suitable for problems with both stiff and non-stiff components
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0      | 0      0      0
    /// 1      | 4γ+2β  1-4γ-2β 0
    /// 1/2    | α₃₁    γ      β
    /// -------|------------------
    ///        | 1/6    1/6    2/3
    /// ```
    /// where:
    /// - β = √3/6 + 1/2 ≈ 0.7886751346
    /// - γ = (-1/8)(√3 + 1) ≈ -0.3416407865
    /// - α₃₁ = 1/2 - β - γ ≈ 0.0529656519
    ///
    /// # References
    /// - Conde, S., et al. (2017). "Implicit and implicit-explicit strong stability preserving Runge-Kutta methods"
    ///
    pub fn esdirk33() -> Self {
        // Parameters (computed in f64 for precision)
        let sqrt3 = 3.0_f64.sqrt();
        let beta = sqrt3 / 6.0 + 0.5;
        let gamma = (-1.0 / 8.0) * (sqrt3 + 1.0);

        let c = [0.0, 1.0, 0.5];
        let a = [
            [0.0, 0.0, 0.0],
            [
                4.0 * gamma + 2.0 * beta,
                1.0 - 4.0 * gamma - 2.0 * beta,
                0.0,
            ],
            [0.5 - beta - gamma, gamma, beta],
        ];
        let b = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];

        let c = c.map(|x| T::from_f64(x).unwrap());
        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());

        ButcherTableau {
            c,
            a,
            b,
            bh: None,
            bi: None,
            er: None,
        }
    }
}

impl<T: Real> ButcherTableau<T, 4> {
    /// ESDIRK3(2)4L[2]SA: 4-stage, 3rd order ESDIRK method with embedded 2nd order
    ///
    /// # Overview
    /// This provides a 4-stage, explicit singly diagonally implicit Runge-Kutta method with:
    /// - Primary order: 3
    /// - Embedded order: 2 (for error estimation)
    /// - Number of stages: 4
    /// - A-stable and B-stable
    ///
    /// # References
    /// - Kennedy, C.A. and Carpenter, M.H. (2003). "Additive Runge-Kutta schemes for convection-diffusion-reaction equations"
    ///
    pub fn esdirk324l2sa() -> Self {
        // Gamma parameter and derived values (computed in f64 for precision)
        let g = 0.43586652150845899941601945;
        let g2 = g * g;
        let g3 = g2 * g;
        let g4 = g3 * g;
        let g5 = g4 * g;

        let c3 = 3.0 / 5.0;

        // Compute coefficients in f64
        let a32 = c3 * (c3 - 2.0 * g) / (4.0 * g);
        let a31 = c3 - g - a32;

        let b2 = (-2.0 + 3.0 * c3 + 6.0 * g * (1.0 - c3)) / (12.0 * g * (c3 - 2.0 * g));
        let b3 = (1.0 - 6.0 * g + 6.0 * g2) / (3.0 * c3 * (c3 - 2.0 * g));
        let b1 = 1.0 - g - b2 - b3;

        // Embedding coefficients
        let d2_term1 = c3 * (-1.0 + 6.0 * g - 24.0 * g3 + 12.0 * g4 - 6.0 * g5)
            / (4.0 * g * (2.0 * g - c3) * (1.0 - 6.0 * g + 6.0 * g2));
        let d2_term2 = (3.0 - 27.0 * g + 68.0 * g2 - 55.0 * g3 + 21.0 * g4 - 6.0 * g5)
            / (2.0 * (2.0 * g - c3) * (1.0 - 6.0 * g + 6.0 * g2));
        let d2 = d2_term1 + d2_term2;

        let d3 = -g * (-2.0 + 21.0 * g - 68.0 * g2 + 79.0 * g3 - 33.0 * g4 + 12.0 * g5)
            / (c3 * (c3 - 2.0 * g) * (1.0 - 6.0 * g + 6.0 * g2));

        let d4 = -3.0 * g2 * (-1.0 + 4.0 * g - 2.0 * g2 + g3) / (1.0 - 6.0 * g + 6.0 * g2);
        let d1 = 1.0 - d2 - d3 - d4;

        let c = [0.0, 2.0 * g, 3.0 / 5.0, 1.0];
        let a = [
            [0.0, 0.0, 0.0, 0.0],
            [g, g, 0.0, 0.0],
            [a31, a32, g, 0.0],
            [b1, b2, b3, g],
        ];
        let b = [b1, b2, b3, g];
        let bh = [d1, d2, d3, d4]; // 2nd order embedding

        let c = c.map(|x| T::from_f64(x).unwrap());
        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let bh = Some(bh.map(|x| T::from_f64(x).unwrap()));

        ButcherTableau {
            c,
            a,
            b,
            bh,
            bi: None,
            er: None,
        }
    }
}
