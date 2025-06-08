use crate::tableau::ButcherTableau;

impl ButcherTableau<2> {
    /// Butcher Tableau for the Lobatto IIIC method of order 2.
    ///
    /// # Overview
    /// This provides a 2-stage, implicit Runge-Kutta method (Lobatto IIIC) with:
    /// - Primary order: 2
    /// - Number of stages: 2
    ///
    /// # Notes
    /// - Lobatto IIIC methods are L-stable.
    /// - They are algebraically stable and thus B-stable, making them suitable for stiff problems.
    /// - This implementation includes two sets of `b` coefficients. The primary `b` coefficients
    ///   are used for the solution, and `bh` can represent alternative coefficients if needed
    ///   (though their specific use here as a second row is characteristic of Lobatto IIIC).
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0   |  1/2  -1/2
    /// 1   |  1/2   1/2
    /// ----|------------
    ///     |  1/2   1/2   (b coefficients)
    ///     |  1     0     (bh coefficients)
    /// ```
    ///
    /// # References
    /// - Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer. (Page 80, Table 5.7)
    pub const fn lobatto_iiic_2() -> Self {
        let mut c = [0.0; 2];
        let mut a = [[0.0; 2]; 2];
        let mut b = [0.0; 2];
        let mut bh = [0.0; 2];

        c[0] = 0.0;
        c[1] = 1.0;

        a[0][0] = 1.0 / 2.0;
        a[0][1] = -1.0 / 2.0;
        a[1][0] = 1.0 / 2.0;
        a[1][1] = 1.0 / 2.0;

        b[0] = 1.0 / 2.0;
        b[1] = 1.0 / 2.0;

        bh[0] = 1.0;
        bh[1] = 0.0;

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
    /// Butcher Tableau for the Lobatto IIIC method of order 4.
    ///
    /// # Overview
    /// This provides a 3-stage, implicit Runge-Kutta method (Lobatto IIIC) with:
    /// - Primary order: 4
    /// - Number of stages: 3
    ///
    /// # Notes
    /// - Lobatto IIIC methods are L-stable.
    /// - They are algebraically stable and thus B-stable, making them suitable for stiff problems.
    /// - This implementation includes two sets of `b` coefficients. The primary `b` coefficients
    ///   are used for the solution, and `bh` represents the second row of coefficients from the tableau.
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0   |  1/6  -1/3   1/6
    /// 1/2 |  1/6   5/12 -1/12
    /// 1   |  1/6   2/3   1/6
    /// ----|------------------
    ///     |  1/6   2/3   1/6   (b coefficients)
    ///     | -1/2   2    -1/2   (bh coefficients)
    /// ```
    ///
    /// # References
    /// - Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer. (Page 80, Table 5.7)
    pub const fn lobatto_iiic_4() -> Self {
        let mut c = [0.0; 3];
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        let mut bh = [0.0; 3];

        c[0] = 0.0;
        c[1] = 1.0 / 2.0;
        c[2] = 1.0;

        a[0][0] = 1.0 / 6.0;
        a[0][1] = -1.0 / 3.0;
        a[0][2] = 1.0 / 6.0;

        a[1][0] = 1.0 / 6.0;
        a[1][1] = 5.0 / 12.0;
        a[1][2] = -1.0 / 12.0;

        a[2][0] = 1.0 / 6.0;
        a[2][1] = 2.0 / 3.0;
        a[2][2] = 1.0 / 6.0;

        b[0] = 1.0 / 6.0;
        b[1] = 2.0 / 3.0;
        b[2] = 1.0 / 6.0;

        bh[0] = -1.0 / 2.0;
        bh[1] = 2.0;
        bh[2] = -1.0 / 2.0;

        Self {
            c,
            a,
            b,
            bh: Some(bh),
            bi: None,
        }
    }
}
