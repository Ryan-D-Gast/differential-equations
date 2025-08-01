//! Kvaerno Diagonally Implicit Runge-Kutta (DIRK) methods

use crate::{tableau::ButcherTableau, traits::Real};

impl<T: Real> ButcherTableau<T, 4> {
    /// Kvaerno(4,2,3): 4-stage, 3rd order DIRK method with embedded 2nd order
    ///
    /// # Overview
    /// This provides a 4-stage, diagonally implicit Runge-Kutta method with:
    /// - Primary order: 3
    /// - Embedded order: 2 (for error estimation)
    /// - Number of stages: 4
    /// - A-stable
    ///
    /// # Notes
    /// - Developed specifically for stiff differential equations
    /// - A-stable method suitable for moderately stiff problems
    /// - The embedded method provides efficient error estimation for adaptive stepping
    /// - All diagonal elements are identical (γ ≈ 0.4358665215), simplifying LU factorization
    /// - Good balance between stability and computational efficiency
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0      | 0      0      0      0
    /// .8717  | .4359  .4359  0      0
    /// 1      | .4906  .0736  .4359  0  
    /// 1      | .3088  1.4906 -1.2352 .4359
    /// -------|-------------------------
    /// b³     | .3088  1.4906 -1.2352 .4359
    /// b²     | .4906  .0736  .4359  0
    /// ```
    /// where γ ≈ 0.4358665215
    ///
    /// # References
    /// - Kvaerno, A. (2004). "Singly diagonally implicit Runge-Kutta methods with an explicit first stage"
    ///
    pub fn kvaerno423() -> Self {
        // Main diagonal entry
        let gamma = 0.4358665215;

        let c = [0.0, 0.871733043, 1.0, 1.0];

        let a = [
            [0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0],
            [0.490563388419108, 0.073570090080892, gamma, 0.0],
            [
                0.308809969973036,
                1.490563388254106,
                -1.235239879727145,
                gamma,
            ],
        ];

        let b = [
            0.308809969973036,
            1.490563388254106,
            -1.235239879727145,
            gamma,
        ];

        let bh = [0.490563388419108, 0.073570090080892, gamma, 0.0];

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

impl<T: Real> ButcherTableau<T, 7> {
    /// Kvaerno(7,4,5): 7-stage, 5th order DIRK method with embedded 4th order
    ///
    /// # Overview
    /// This provides a 7-stage, diagonally implicit Runge-Kutta method with:
    /// - Primary order: 5
    /// - Embedded order: 4 (for error estimation)
    /// - Number of stages: 7
    /// - A-stable and B-stable
    ///
    /// # Notes
    /// - High-order method designed for stiff differential equations requiring high accuracy
    /// - A-stable and B-stable for excellent stability properties
    /// - The embedded 4th order method provides accurate error estimation for adaptive control
    /// - All diagonal elements are identical (γ = 0.26), enabling efficient LU factorization reuse
    /// - Particularly effective for smooth solutions where high accuracy is required
    /// - Higher computational cost per step but fewer steps needed due to high order
    ///
    /// # Butcher Tableau
    /// ```text
    /// 0      | 0      0      0      0      0      0      0
    /// .52    | .26    .26    0      0      0      0      0
    /// 1.2303 | .13    .8403  .26    0      0      0      0
    /// .8958  | .2237  .4768  -.0647 .26    0      0      0
    /// .4364  | .1665  .1045  .0363  -.1309 .26    0      0
    /// 1      | .1386  .0000  -.0425 .0245  .6194  .26    0
    /// 1      | .1366  .0000  -.0550 -.0412 .6299  .0696  .26
    /// -------|--------------------------------------------
    /// b⁵     | .1366  .0000  -.0550 -.0412 .6299  .0696  .26
    /// b⁴     | .1386  .0000  -.0425 .0245  .6194  .26    0
    /// ```
    /// where γ = 0.26 exactly
    ///
    /// # References
    /// - Kvaerno, A. (2004). "Singly diagonally implicit Runge-Kutta methods with an explicit first stage"
    ///
    pub fn kvaerno745() -> Self {
        // Main diagonal entry
        let gamma = 0.26;

        let c = [
            0.0,
            0.52,
            1.230333209967908,
            0.895765984350076,
            0.436393609858648,
            1.0,
            1.0,
        ];

        let a = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.13, 0.840_333_209_967_908_1, gamma, 0.0, 0.0, 0.0, 0.0],
            [
                0.22371961478320505,
                0.476_755_323_197_997,
                -0.06470895363112615,
                gamma,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.16648564323248321,
                0.104_500_188_415_917_2,
                0.03631482272098715,
                -0.13090704451073998,
                gamma,
                0.0,
                0.0,
            ],
            [
                0.13855640231268224,
                0.0,
                -0.04245337201752043,
                0.02446657898003141,
                0.619_430_390_724_806_8,
                gamma,
                0.0,
            ],
            [
                0.13659751177640291,
                0.0,
                -0.05496908796538376,
                -0.04118626728321046,
                0.629_933_048_990_164,
                0.06962479448202728,
                gamma,
            ],
        ];

        let b = [
            0.13659751177640291,
            0.0,
            -0.05496908796538376,
            -0.04118626728321046,
            0.629_933_048_990_164,
            0.06962479448202728,
            gamma,
        ];

        let bh = [
            0.13855640231268224,
            0.0,
            -0.04245337201752043,
            0.02446657898003141,
            0.619_430_390_724_806_8,
            gamma,
            0.0,
        ];

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
