//! Kvaerno Diagonally Implicit Runge-Kutta (DIRK) methods

use crate::{
    tableau::ButcherTableau,
    traits::Real
};

impl<T: Real> ButcherTableau<T, 4> {
    /// Kvaerno(4,2,3)-ESDIRK: 4-stage, 3rd order ESDIRK method with 2nd order embedding
    /// 
    /// High-quality 3rd order ESDIRK method with embedded 2nd order method for adaptive stepping.
    /// L-stable and A-stable with good stability properties.
    /// 
    pub fn kvaerno423() -> Self {
        // Main diagonal entry
        let gamma = 0.4358665215;

        let c = [
            0.0,
            0.871733043,
            1.0,
            1.0
        ];

        let a = [
            [0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0],
            [0.490563388419108, 0.073570090080892, gamma, 0.0],
            [0.308809969973036, 1.490563388254106, -1.235239879727145, gamma]
        ];

        let b = [
            0.308809969973036,
            1.490563388254106, 
            -1.235239879727145,
            gamma
        ];

        let bh = [
            0.490563388419108,
            0.073570090080892,
            gamma,
            0.0
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

impl<T: Real> ButcherTableau<T, 7> {
    /// Kvaerno(7,4,5)-ESDIRK: 7-stage, 5th order ESDIRK method with 4th order embedding
    /// 
    /// High-order ESDIRK method with excellent stability properties. Suitable for 
    /// problems requiring high accuracy and strong stability.
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
            1.0
        ];

        let a = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [gamma, gamma, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.13, 0.84033320996790809, gamma, 0.0, 0.0, 0.0, 0.0],
            [0.22371961478320505, 0.47675532319799699, -0.06470895363112615, gamma, 0.0, 0.0, 0.0],
            [0.16648564323248321, 0.10450018841591720, 0.03631482272098715, -0.13090704451073998, gamma, 0.0, 0.0],
            [0.13855640231268224, 0.0, -0.04245337201752043, 0.02446657898003141, 0.61943039072480676, gamma, 0.0],
            [0.13659751177640291, 0.0, -0.05496908796538376, -0.04118626728321046, 0.62993304899016403, 0.06962479448202728, gamma]
        ];

        let b = [
            0.13659751177640291,
            0.0,
            -0.05496908796538376,
            -0.04118626728321046,
            0.62993304899016403,
            0.06962479448202728,
            gamma
        ];

        let bh = [
            0.13855640231268224,
            0.0,
            -0.04245337201752043,
            0.02446657898003141,
            0.61943039072480676,
            gamma,
            0.0
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