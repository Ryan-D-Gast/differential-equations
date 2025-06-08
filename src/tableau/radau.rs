//! Radau Implicit Runge-Kutta methods.

use crate::tableau::ButcherTableau;

const SQRT_6: f64 = 2.449489743;

impl ButcherTableau<2> {
    /// Butcher Tableau for the Radau IIA method of order 3.
    ///
    /// # Overview
    /// This provides a 2-stage, implicit Runge-Kutta method (Radau IIA) with:
    /// - Primary order: 3
    /// - Embedded order: None (this implementation does not include an embedded error estimate)
    /// - Number of stages: 2
    ///
    /// # Interpolation
    /// - This standard Radau IIA order 3 implementation does not provide coefficients for dense output (interpolation).
    ///
    /// # Notes
    /// - Radau IIA methods are A-stable and L-stable, making them suitable for stiff differential equations.
    /// - Being implicit, they generally require solving a system of algebraic equations at each step.
    ///
    /// # Butcher Tableau
    /// ```text
    /// 1/3 |  5/12  -1/12
    /// 1   |  3/4    1/4
    /// ----|---------------
    ///     |  3/4    1/4
    /// ```
    ///
    /// # References
    /// - Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.
    /// - Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer.
    pub const fn radau_iia_3() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];

        c[0] = 1.0 / 3.0;
        c[1] = 1.0;

        a[0][0] = 5.0 / 12.0;
        a[0][1] = -1.0 / 12.0;
        a[1][0] = 3.0 / 4.0;
        a[1][1] = 1.0 / 4.0;

        b[0] = 3.0 / 4.0;
        b[1] = 1.0 / 4.0;

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}

impl ButcherTableau<3> {
    /// Butcher Tableau for the Radau IIA method of order 5.
    ///
    /// # Overview
    /// This provides a 3-stage, implicit Runge-Kutta method (Radau IIA) with:
    /// - Primary order: 5
    /// - Embedded order: None (this implementation does not include an embedded error estimate)
    /// - Number of stages: 3
    ///
    /// # Interpolation
    /// - This standard Radau IIA order 5 implementation does not provide coefficients for dense output (interpolation).
    ///
    /// # Notes
    /// - Radau IIA methods are A-stable and L-stable, making them highly suitable for stiff differential equations.
    /// - Being implicit, they generally require solving a system of algebraic equations at each step.
    /// - The `c` values are the roots of `P_s(2x-1) - P_{s-1}(2x-1) = 0` where `P_s` is the s-th Legendre polynomial.
    ///   For s=3, the c values are `(2/5 - sqrt(6)/10)`, `(2/5 + sqrt(6)/10)`, and `1`.
    ///
    /// # Butcher Tableau
    /// ```text
    /// c1 | a11 a12 a13
    /// c2 | a21 a22 a23
    /// c3 | a31 a32 a33
    /// ---|------------
    ///    | b1  b2  b3
    /// ```
    /// Where:
    /// c1 = 2/5 - sqrt(6)/10
    /// c2 = 2/5 + sqrt(6)/10
    /// c3 = 1
    ///
    /// a11 = 11/45 - 7*sqrt(6)/360
    /// a12 = 37/225 - 169*sqrt(6)/1800
    /// a13 = -2/225 + sqrt(6)/75
    ///
    /// a21 = 37/225 + 169*sqrt(6)/1800
    /// a22 = 11/45 + 7*sqrt(6)/360
    /// a23 = -2/225 - sqrt(6)/75
    ///
    /// a31 = 4/9 - sqrt(6)/36
    /// a32 = 4/9 + sqrt(6)/36
    /// a33 = 1/9
    ///
    /// b1 = 4/9 - sqrt(6)/36
    /// b2 = 4/9 + sqrt(6)/36
    /// b3 = 1/9
    ///
    /// # References
    /// - Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.
    /// - Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer.
    pub const fn radau_iia_5() -> Self {
        let mut c = [0.0; 3];
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];

        c[0] = 2.0 / 5.0 - SQRT_6 / 10.0;
        c[1] = 2.0 / 5.0 + SQRT_6 / 10.0;
        c[2] = 1.0;

        a[0][0] = 11.0 / 45.0 - 7.0 * SQRT_6 / 360.0;
        a[0][1] = 37.0 / 225.0 - 169.0 * SQRT_6 / 1800.0;
        a[0][2] = -2.0 / 225.0 + SQRT_6 / 75.0;

        a[1][0] = 37.0 / 225.0 + 169.0 * SQRT_6 / 1800.0;
        a[1][1] = 11.0 / 45.0 + 7.0 * SQRT_6 / 360.0;
        a[1][2] = -2.0 / 225.0 - SQRT_6 / 75.0;

        a[2][0] = 4.0 / 9.0 - SQRT_6 / 36.0;
        a[2][1] = 4.0 / 9.0 + SQRT_6 / 36.0;
        a[2][2] = 1.0 / 9.0;

        b[0] = 4.0 / 9.0 - SQRT_6 / 36.0;
        b[1] = 4.0 / 9.0 + SQRT_6 / 36.0;
        b[2] = 1.0 / 9.0;

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}