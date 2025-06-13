//! Classic or typical Runge-Kutta methods without unique properties

use crate::{
    tableau::ButcherTableau,
    traits::Real,
};

impl<T: Real> ButcherTableau<T, 4> {
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
    pub fn rk4() -> Self {
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

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
    
    /// Three-Eighths Rule 4th order method.
    ///
    /// # Overview
    /// This provides a 4-stage, explicit Runge-Kutta method with:
    /// - Primary order: 4
    /// - Embedded order: None (no embedded error estimate)
    /// - Number of stages: 4
    ///
    /// # Notes
    /// - The primary advantage of this method is that almost all of the error coefficients
    ///   are smaller than in the classic RK4 method, but it requires slightly more FLOPs
    ///   (floating-point operations) per time step.
    ///
    /// # Butcher Tableau
    /// ```
    /// 0   |
    /// 1/3 | 1/3
    /// 2/3 | -1/3 1
    /// 1   | 1   -1   1
    /// ----|--------------------
    ///     | 1/8 3/8 3/8 1/8
    /// ```
    ///
    /// # References
    /// - Butcher, J.C. (2008). "Numerical Methods for Ordinary Differential Equations".
    pub fn three_eighths() -> Self {
        let mut c = [0.0; 4];
        let mut a = [[0.0; 4]; 4];
        let mut b = [0.0; 4];

        c[0] = 0.0;
        c[1] = 1.0/3.0;
        c[2] = 2.0/3.0;
        c[3] = 1.0;

        a[1][0] = 1.0/3.0;
        a[2][0] = -1.0/3.0;
        a[2][1] = 1.0;
        a[3][0] = 1.0;
        a[3][1] = -1.0;
        a[3][2] = 1.0;

        b[0] = 1.0/8.0;
        b[1] = 3.0/8.0;
        b[2] = 3.0/8.0;
        b[3] = 1.0/8.0;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}

impl<T: Real> ButcherTableau<T, 2> {
    /// Midpoint method (2nd order Runge-Kutta).
    ///
    /// # Overview
    /// This provides a 2-stage, explicit Runge-Kutta method with:
    /// - Primary order: 2
    /// - Embedded order: None
    /// - Number of stages: 2
    ///
    /// # Butcher Tableau
    /// ```
    /// 0   |
    /// 1/2 | 1/2
    /// ----|--------
    ///     | 0   1
    /// ```
    ///
    /// # References
    /// - Butcher, J.C. (2008). "Numerical Methods for Ordinary Differential Equations".
    pub fn midpoint() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];

        c[0] = 0.0;
        c[1] = 0.5;

        a[1][0] = 0.5;

        b[0] = 0.0;
        b[1] = 1.0;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }

    /// Heun's method (2nd order Runge-Kutta).
    ///
    /// # Overview
    /// This provides a 2-stage, explicit Runge-Kutta method with:
    /// - Primary order: 2
    /// - Embedded order: None
    /// - Number of stages: 2
    ///
    /// # Butcher Tableau
    /// ```
    /// 0   |
    /// 1   | 1
    /// ----|--------
    ///     | 1/2 1/2
    /// ```
    ///
    /// # References
    /// - Heun, K. (1900). "Neue Methoden zur approximativen Integration der Differentialgleichungen einer unabhängigen Veränderlichen".
    pub fn heun() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];

        c[0] = 0.0;
        c[1] = 1.0;

        a[1][0] = 1.0;

        b[0] = 0.5;
        b[1] = 0.5;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }

    /// Ralston's method (2nd order Runge-Kutta).
    ///
    /// # Overview
    /// This provides a 2-stage, explicit Runge-Kutta method with:
    /// - Primary order: 2 
    /// - Embedded order: None
    /// - Number of stages: 2
    /// 
    /// # Butcher Tableau
    /// ```
    /// 0   |
    /// 2/3 | 2/3
    /// ----|--------
    ///     | 1/4 3/4
    /// ```
    ///
    /// # References
    /// - Ralston, A. (1962). "Runge-Kutta Methods with Minimum Error Bounds".
    pub fn ralston() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];

        c[0] = 0.0;
        c[1] = 2.0/3.0;

        a[1][0] = 2.0/3.0;

        b[0] = 1.0/4.0;
        b[1] = 3.0/4.0;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}

impl<T: Real> ButcherTableau<T, 1> {
    /// Euler's method (1st order Runge-Kutta).
    ///
    /// # Overview
    /// This provides a 1-stage, explicit Runge-Kutta method with:
    /// - Primary order: 1
    /// - Embedded order: None
    /// - Number of stages: 1
    ///
    /// # Butcher Tableau
    /// ```
    /// 0 | 
    /// --|--
    ///   | 1
    /// ```
    ///
    /// # References
    /// - Euler, L. (1768). "Institutionum calculi integralis".
    pub fn euler() -> Self {
        let c = [T::zero()];
        let a = [[T::zero()]];
        let b = [T::one()];

        Self {
            c,
            a,
            b,
            bh: None,
            bi: None,
        }
    }
}

// Implementations for methods with embedded error estimators (adaptive methods)
impl<T: Real> ButcherTableau<T, 6> {
    /// Runge-Kutta-Fehlberg 4(5) method (RKF45).
    ///
    /// # Overview
    /// This provides a 6-stage, explicit Runge-Kutta method with:
    /// - Primary order: 5
    /// - Embedded order: 4 (for error estimation)
    /// - Number of stages: 6
    ///
    /// # Notes
    /// - RKF45 is a widely used adaptive step size method that provides a good balance
    ///   between accuracy and computational efficiency.
    /// - It uses the difference between 4th and 5th order approximations to estimate error.
    ///
    /// # Butcher Tableau
    /// ```
    /// 0      |
    /// 1/4    | 1/4
    /// 3/8    | 3/32         9/32
    /// 12/13  | 1932/2197    -7200/2197  7296/2197
    /// 1      | 439/216      -8          3680/513    -845/4104
    /// 1/2    | -8/27        2           -3544/2565  1859/4104   -11/40
    /// -------|---------------------------------------------------------------
    ///        | 16/135       0           6656/12825  28561/56430 -9/50  2/55  (5th)
    ///        | 25/216       0           1408/2565   2197/4104   -1/5   0     (4th)
    /// ```
    ///
    /// # References
    /// - Fehlberg, E. (1969). "Low-order classical Runge-Kutta formulas with step size control and their application to some heat transfer problems".
    pub fn rkf45() -> Self {
        let mut c = [0.0; 6];
        let mut a = [[0.0; 6]; 6];
        let mut b = [0.0; 6];
        let mut bh = [0.0; 6];

        c[0] = 0.0;
        c[1] = 1.0/4.0;
        c[2] = 3.0/8.0;
        c[3] = 12.0/13.0;
        c[4] = 1.0;
        c[5] = 1.0/2.0;

        a[1][0] = 1.0/4.0;
        a[2][0] = 3.0/32.0;
        a[2][1] = 9.0/32.0;
        a[3][0] = 1932.0/2197.0;
        a[3][1] = -7200.0/2197.0;
        a[3][2] = 7296.0/2197.0;
        a[4][0] = 439.0/216.0;
        a[4][1] = -8.0;
        a[4][2] = 3680.0/513.0;
        a[4][3] = -845.0/4104.0;
        a[5][0] = -8.0/27.0;
        a[5][1] = 2.0;
        a[5][2] = -3544.0/2565.0;
        a[5][3] = 1859.0/4104.0;
        a[5][4] = -11.0/40.0;

        b[0] = 16.0/135.0;
        b[1] = 0.0;
        b[2] = 6656.0/12825.0;
        b[3] = 28561.0/56430.0;
        b[4] = -9.0/50.0;
        b[5] = 2.0/55.0;

        bh[0] = 25.0/216.0;
        bh[1] = 0.0;
        bh[2] = 1408.0/2565.0;
        bh[3] = 2197.0/4104.0;
        bh[4] = -1.0/5.0;
        bh[5] = 0.0;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let bh = bh.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
        }
    }

    /// Cash-Karp 4(5) method.
    ///
    /// # Overview
    /// This provides a 6-stage, explicit Runge-Kutta method with:
    /// - Primary order: 5
    /// - Embedded order: 4 (for error estimation)
    /// - Number of stages: 6
    ///
    /// # Notes
    /// - The Cash-Karp method is a variant of Runge-Kutta methods with embedded error estimation
    ///   that often provides better accuracy than RKF45 for some problem types.
    ///
    /// # Butcher Tableau
    /// ```
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 3/5    | 3/10         -9/10       6/5
    /// 1      | -11/54       5/2         -70/27      35/27
    /// 7/8    | 1631/55296   175/512     575/13824   44275/110592 253/4096
    /// -------|---------------------------------------------------------------
    ///        | 37/378       0           250/621     125/594     0      512/1771  (5th)
    ///        | 2825/27648   0           18575/48384 13525/55296 277/14336 1/4    (4th)
    /// ```
    ///
    /// # References
    /// - Cash, J.R., Karp, A.H. (1990). "A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides".
    pub fn cash_karp() -> Self {
        let mut c = [0.0; 6];
        let mut a = [[0.0; 6]; 6];
        let mut b = [0.0; 6];
        let mut bh = [0.0; 6];

        c[0] = 0.0;
        c[1] = 1.0/5.0;
        c[2] = 3.0/10.0;
        c[3] = 3.0/5.0;
        c[4] = 1.0;
        c[5] = 7.0/8.0;

        a[1][0] = 1.0/5.0;
        a[2][0] = 3.0/40.0;
        a[2][1] = 9.0/40.0;
        a[3][0] = 3.0/10.0;
        a[3][1] = -9.0/10.0;
        a[3][2] = 6.0/5.0;
        a[4][0] = -11.0/54.0;
        a[4][1] = 5.0/2.0;
        a[4][2] = -70.0/27.0;
        a[4][3] = 35.0/27.0;
        a[5][0] = 1631.0/55296.0;
        a[5][1] = 175.0/512.0;
        a[5][2] = 575.0/13824.0;
        a[5][3] = 44275.0/110592.0;
        a[5][4] = 253.0/4096.0;

        b[0] = 37.0/378.0;
        b[1] = 0.0;
        b[2] = 250.0/621.0;
        b[3] = 125.0/594.0;
        b[4] = 0.0;
        b[5] = 512.0/1771.0;

        bh[0] = 2825.0/27648.0;
        bh[1] = 0.0;
        bh[2] = 18575.0/48384.0;
        bh[3] = 13525.0/55296.0;
        bh[4] = 277.0/14336.0;
        bh[5] = 1.0/4.0;

        let a = a.map(|row| row.map(|x| T::from_f64(x).unwrap()));
        let b = b.map(|x| T::from_f64(x).unwrap());
        let bh = bh.map(|x| T::from_f64(x).unwrap());
        let c = c.map(|x| T::from_f64(x).unwrap());

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
        }
    }
}