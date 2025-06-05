use crate::tableau::{
    ButcherTableau,
    NumicalMethodType,
};

impl ButcherTableau<5, 4, 7, 1, 7, 7> {
    /// Dormand-Prince 5(4) Tableau with dense output interpolation.
    ///
    /// # Overview
    /// This provides a 7-stage Runge-Kutta method with:
    /// - Primary order: 5
    /// - Embedded order: 4 (for error estimation)
    /// - Number of stages: 7 primary + 0 additional stages for interpolation
    /// - Built-in dense output of order 4
    ///
    /// # Efficiency
    /// The DOPRI5 method is popular due to its efficient balance between accuracy and 
    /// computational cost. It is particularly good for non-stiff problems.
    ///
    /// # Interpolation
    /// - The method provides a 4th-order interpolant using the existing 7 stages
    /// - The interpolant has continuous first derivatives
    /// - The interpolant uses the values of the stages to construct a polynomial
    ///   that allows evaluation at any point within the integration step
    ///
    /// # Notes
    /// - The method was developed by Dormand and Prince in 1980
    /// - It is one of the most widely used Runge-Kutta methods and is implemented
    ///   in many software packages for solving ODEs
    /// - The DOPRI5 method is a member of the Dormand-Prince family of embedded
    ///   Runge-Kutta methods
    ///
    /// # References
    /// - Dormand, J. R. & Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
    ///   Journal of Computational and Applied Mathematics, 6(1), pp. 19-26
    /// - Hairer, E., Nørsett, S. P. & Wanner, G. (1993), "Solving Ordinary Differential Equations I: 
    ///   Nonstiff Problems", Springer Series in Computational Mathematics, Vol. 8, Springer-Verlag
    pub const fn dopri5() -> Self {
        let mut c = [0.0; 7];
        let mut a = [[0.0; 7]; 7];
        let mut b = [0.0; 7];
        let mut bh = [0.0; 7];
        let mut bi4 = [[0.0; 7]; 1];

        c[0] = 0.0;
        c[1] = 0.2;
        c[2] = 0.3;
        c[3] = 0.8;
        c[4] = 8.0 / 9.0;
        c[5] = 1.0;
        c[6] = 1.0;

        a[1][0] = 0.2;

        a[2][0] = 3.0 / 40.0;
        a[2][1] = 9.0 / 40.0;

        a[3][0] = 44.0 / 45.0;
        a[3][1] = -56.0 / 15.0;
        a[3][2] = 32.0 / 9.0;

        a[4][0] = 19372.0 / 6561.0;
        a[4][1] = -25360.0 / 2187.0;
        a[4][2] = 64448.0 / 6561.0;
        a[4][3] = -212.0 / 729.0;

        a[5][0] = 9017.0 / 3168.0;
        a[5][1] = -355.0 / 33.0;
        a[5][2] = 46732.0 / 5247.0;
        a[5][3] = 49.0 / 176.0;
        a[5][4] = -5103.0 / 18656.0;

        a[6][0] = 35.0 / 384.0;
        a[6][1] = 0.0;
        a[6][2] = 500.0 / 1113.0;
        a[6][3] = 125.0 / 192.0;
        a[6][4] = -2187.0 / 6784.0;
        a[6][5] = 11.0 / 84.0;

        b[0] = 35.0 / 384.0;
        b[1] = 0.0;
        b[2] = 500.0 / 1113.0;
        b[3] = 125.0 / 192.0;
        b[4] = -2187.0 / 6784.0;
        b[5] = 11.0 / 84.0;
        b[6] = 0.0;

        bh[0] = 71.0 / 57600.0;
        bh[1] = 0.0;
        bh[2] = -71.0 / 16695.0;
        bh[3] = 71.0 / 1920.0;
        bh[4] = -17253.0 / 339200.0;
        bh[5] = 22.0 / 525.0;
        bh[6] = 1.0 / 40.0;

        bi4[0][0] = -12715105075.0 / 11282082432.0;
        bi4[0][1] = 0.0;
        bi4[0][2] = 87487479700.0 / 32700410799.0;
        bi4[0][3] = -10690763975.0 / 1880347072.0;
        bi4[0][4] = 701980252875.0 / 199316789632.0;
        bi4[0][5] = -1453857185.0 / 822651844.0;
        bi4[0][6] = 69997945.0 / 29380423.0;

        ButcherTableau {
            method: NumicalMethodType::DormandPrince,
            c,
            a,
            b,
            bh: Some(bh),
            bi: Some(bi4),
        }
    }
}