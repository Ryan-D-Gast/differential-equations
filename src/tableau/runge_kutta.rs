//! Classic or typical Runge-Kutta methods without unique properties

use crate::tableau::ButcherTableau;

impl ButcherTableau<4> {
    /// Classic Runge-Kutta 4th order method (RK4).
    ///
    /// # Overview
    /// This provides a 4-stage, explicit Runge-Kutta method with:
    /// - Primary order: 4
    /// - Embedded order: None (this implementation does not include an embedded error estimate)
    /// - Number of stages: 4
    ///
    /// # Interpolation
    /// - This standard RK4 implementation does not provide coefficients for dense output (interpolation).
    ///
    /// # Notes
    /// - RK4 is a widely used, general-purpose method known for its balance of accuracy and
    ///   computational efficiency for many problems.
    /// - It is a fixed-step method as presented here, lacking an embedded formula for adaptive
    ///   step-size control. For adaptive step sizes, a method with an error estimate (e.g., RKF45)
    ///   would be required.
    ///
    /// # Butcher Tableau
    /// ```
    /// 0   |
    /// 1/2 | 1/2
    /// 1/2 | 0   1/2
    /// 1   | 0   0   1
    /// ----|--------------------
    ///     | 1/6 1/3 1/3 1/6
    /// ```
    ///
    /// # References
    /// - Kutta, W. (1901). "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen". *Zeitschrift für Mathematik und Physik*, 46, 435-453.
    /// - Runge, C. (1895). "Über die numerische Auflösung von Differentialgleichungen". *Mathematische Annalen*, 46, 167-178.
    pub const fn rk4() -> Self {
        let mut c = [0.0; 4];
        let mut a = [[0.0; 4]; 4];
        let mut b = [0.0; 4];

        c[0] = 0.0;
        c[1] = 0.5;
        c[2] = 0.5;
        c[3] = 1.0;

        a[1][0] = 0.5;
        a[2][1] = 0.5;
        a[3][2] = 1.0;

        b[0] = 1.0 / 6.0;
        b[1] = 1.0 / 3.0;
        b[2] = 1.0 / 3.0;
        b[3] = 1.0 / 6.0;

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}