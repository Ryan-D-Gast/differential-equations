//! Gauss-Legendre Implicit Runge-Kutta Methods

use crate::tableau::ButcherTableau;

const SQRT_3: f64 = 1.732050808;

impl ButcherTableau<2> {
    /// Butcher Tableau for the Gauss-Legendre method of order 4.
    ///
    /// # Overview
    /// This provides a 2-stage, implicit Runge-Kutta method (Gauss-Legendre) with:
    /// - Primary order: 4
    /// - Number of stages: 2
    ///
    /// # Notes
    /// - Gauss-Legendre methods are A-stable and symmetric.
    /// - They are highly accurate for their number of stages.
    /// - The `c` values are the roots of the Legendre polynomial P_s(2x-1) = 0.
    /// - This implementation includes two sets of `b` coefficients. The primary `b` coefficients
    ///   are used for the solution, and `bh` can represent alternative coefficients (often related to error estimation or specific properties).
    ///
    /// # Butcher Tableau
    /// ```text
    /// (1/2 - sqrt(3)/6) |  1/4                  1/4 - sqrt(3)/6
    /// (1/2 + sqrt(3)/6) |  1/4 + sqrt(3)/6      1/4
    /// -------------------|---------------------------------------
    ///                    |  1/2                  1/2
    ///                    |  1/2 + sqrt(3)/2      1/2 - sqrt(3)/2  (bh coefficients)
    /// ```
    ///
    /// # References
    /// - Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer. (Page 200, Table 4.5)
    pub const fn gauss_legendre_4() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];
        let mut bh = [0.0; 2];

        let sqrt3_6 = SQRT_3 / 6.0;
        let sqrt3_2 = SQRT_3 / 2.0;

        c[0] = 0.5 - sqrt3_6;
        c[1] = 0.5 + sqrt3_6;

        a[0][0] = 0.25;
        a[0][1] = 0.25 - sqrt3_6;
        a[1][0] = 0.25 + sqrt3_6;
        a[1][1] = 0.25;

        b[0] = 0.5;
        b[1] = 0.5;

        bh[0] = 0.5 + sqrt3_2;
        bh[1] = 0.5 - sqrt3_2;

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
        }
    }
}

impl ButcherTableau<3> {
    /// Butcher Tableau for the Gauss-Legendre method of order 6.
    ///
    /// # Overview
    /// This provides a 3-stage, implicit Runge-Kutta method (Gauss-Legendre) with:
    /// - Primary order: 6
    /// - Number of stages: 3
    ///
    /// # Notes
    /// - Gauss-Legendre methods are A-stable and symmetric.
    /// - They are highly accurate for their number of stages.
    /// - The `c` values are the roots of the Legendre polynomial P_s(2x-1) = 0.
    /// - This implementation includes two sets of `b` coefficients. The primary `b` coefficients
    ///   are used for the solution, and `bh` can represent alternative coefficients.
    ///
    /// # Butcher Tableau
    /// ```text
    /// (1/2 - sqrt(15)/10) |  5/36                2/9 - sqrt(15)/15    5/36 - sqrt(15)/30
    ///  1/2                |  5/36 + sqrt(15)/24  2/9                  5/36 - sqrt(15)/24
    /// (1/2 + sqrt(15)/10) |  5/36 + sqrt(15)/30  2/9 + sqrt(15)/15    5/36
    /// --------------------|-------------------------------------------------------------
    ///                     |  5/18              4/9                  5/18
    ///                     | -5/6               8/3                 -5/6                (bh coefficients)
    /// ```
    ///
    /// # References
    /// - Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer. (Page 200, Table 4.5)
    pub const fn gauss_legendre_6() -> Self {
        let mut c = [0.0; 3];
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        let mut bh = [0.0; 3];

        let sqrt_15: f64 = 3.872983346207417;
        let sqrt15_10 = sqrt_15 / 10.0;
        let sqrt15_15 = sqrt_15 / 15.0;
        let sqrt15_24 = sqrt_15 / 24.0;
        let sqrt15_30 = sqrt_15 / 30.0;

        c[0] = 0.5 - sqrt15_10;
        c[1] = 0.5;
        c[2] = 0.5 + sqrt15_10;

        a[0][0] = 5.0 / 36.0;
        a[0][1] = 2.0 / 9.0 - sqrt15_15;
        a[0][2] = 5.0 / 36.0 - sqrt15_30;

        a[1][0] = 5.0 / 36.0 + sqrt15_24;
        a[1][1] = 2.0 / 9.0;
        a[1][2] = 5.0 / 36.0 - sqrt15_24;

        a[2][0] = 5.0 / 36.0 + sqrt15_30;
        a[2][1] = 2.0 / 9.0 + sqrt15_15;
        a[2][2] = 5.0 / 36.0;

        b[0] = 5.0 / 18.0;
        b[1] = 4.0 / 9.0;
        b[2] = 5.0 / 18.0;

        bh[0] = -5.0 / 6.0;
        bh[1] = 8.0 / 3.0;
        bh[2] = -5.0 / 6.0;

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
        }
    }
}
