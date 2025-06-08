//! Bogacki-Shampine Runge-Kutta method.

use crate::tableau::ButcherTableau;

impl ButcherTableau<4> {
    /// Butcher Tableau for the Bogacki-Shampine method.
    ///
    /// # Overview
    /// This provides a 4-stage, explicit Runge-Kutta method with:
    /// - Primary order: 3 (given by `b` coefficients)
    /// - Embedded order: 2 (given by `bh` coefficients for error estimation)
    /// - Number of stages: 4
    /// - FSAL (First Same As Last) property: The last stage `k_3` (using 0-indexed stages)
    ///   evaluated at `t_n + h` can be used as the first stage `k_0` of the next step.
    ///
    /// # Interpolation
    /// Coefficients `bi` are provided for a cubic Hermite interpolant of order 3.
    /// The interpolated solution at `t_n + θh` is given by:
    /// `y_interp(θ) = y_n + h * Σ_{i=0..3} (k_i * Σ_{j=0..2} bi[j][i] * θ^(j+1))`
    /// This uses the first three stage derivatives `k_0, k_1, k_2`.
    ///
    /// # Notes
    /// - This method is well-suited for problems requiring moderate accuracy.
    /// - It is used as the basis for MATLAB's `ode23` solver.
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0   |
    /// 1/2 | 1/2
    /// 3/4 | 0    3/4
    /// 1   | 2/9  1/3  4/9
    /// ----|--------------------
    /// b   | 2/9  1/3  4/9  0      (3rd order solution)
    /// bh  |7/24  1/4  1/3  1/8    (2nd order solution for error)
    /// ```
    ///
    /// # References
    /// - Bogacki, P., & Shampine, L. F. (1989). "A 3(2) pair of Runge-Kutta formulas". *Applied Mathematics Letters*, 2(4), 321-325.
    /// - Shampine, L. F. (1994). *Numerical solution of ordinary differential equations*. Chapman & Hall. (Discusses the interpolant)
    pub const fn bogacki_shampine() -> Self {
        let mut c = [0.0; 4];
        let mut a = [[0.0; 4]; 4];
        let mut b = [0.0; 4];
        let mut bh = [0.0; 4];
        // let mut bi = [[0.0; 4]; 4];

        c[0] = 0.0;
        c[1] = 1.0 / 2.0;
        c[2] = 3.0 / 4.0;
        c[3] = 1.0;

        a[1][0] = 1.0 / 2.0;
        a[2][1] = 3.0 / 4.0;
        a[3][0] = 2.0 / 9.0;
        a[3][1] = 1.0 / 3.0;
        a[3][2] = 4.0 / 9.0;

        b[0] = 2.0 / 9.0;
        b[1] = 1.0 / 3.0;
        b[2] = 4.0 / 9.0;
        b[3] = 0.0;

        bh[0] = 7.0 / 24.0;
        bh[1] = 1.0 / 4.0;
        bh[2] = 1.0 / 3.0;
        bh[3] = 1.0 / 8.0;

        /* THIS NEEDS REVIEWED
        // bi matrix (interpolation coefficients)
        // Interpolant: y_n + h * [ (bi[0][0]θ + bi[1][0]θ^2 + bi[2][0]θ^3)k_0 +
        //                          (bi[0][1]θ + bi[1][1]θ^2 + bi[2][1]θ^3)k_1 +
        //                          (bi[0][2]θ + bi[1][2]θ^2 + bi[2][2]θ^3)k_2 ]
        // (k_0, k_1, k_2 are the first three stage derivatives. k_3 is not used by this interpolant)

        // Coefficients for k_0 (1st stage derivative)
        bi[0][0] = 1.0;          // θ^1
        bi[1][0] = -3.0 / 2.0;   // θ^2
        bi[2][0] = 2.0 / 3.0;    // θ^3
        // bi[3][0] = 0.0;       // θ^4 (not used for cubic)

        // Coefficients for k_1 (2nd stage derivative)
        // bi[0][1] = 0.0;       // θ^1
        bi[1][1] = 2.0;          // θ^2
        bi[2][1] = -4.0 / 3.0;   // θ^3
        // bi[3][1] = 0.0;       // θ^4

        // Coefficients for k_2 (3rd stage derivative)
        // bi[0][2] = 0.0;       // θ^1
        bi[1][2] = -1.0 / 2.0;   // θ^2
        bi[2][2] = 2.0 / 3.0;    // θ^3
        // bi[3][2] = 0.0;       // θ^4
        
        // Coefficients for k_3 (4th stage derivative) are all zero as it's not used in the interpolant.
        // bi[j][3] = 0.0 for j=0..3
        */

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
            //bi: Some(bi),
        }
    }
}
