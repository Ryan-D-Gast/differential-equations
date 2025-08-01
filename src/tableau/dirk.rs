//! Diagonally Implicit Runge-Kutta (DIRK) tableau

use crate::{
    tableau::ButcherTableau,
    traits::Real
};

impl<T: Real> ButcherTableau<T, 2> {
    /// SDIRK-2-1: 2-stage, 2nd order SDIRK method with 1st order embedding
    /// 
    /// This is a simple 2-stage singly diagonally implicit method with
    /// A-stability and an embedded 1st order method for error estimation.
    /// Good for basic adaptive stepping.
    /// 
    pub fn sdirk21() -> Self {
        let c = [1.0, 0.0];
        let a = [
            [1.0, 0.0],
            [-1.0, 1.0]
        ];
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
    /// This method pairs with SSPRK(3,3)-Shu-Osher-ERK to make a 3rd order IMEX method.
    /// Has an explicit first stage (ESDIRK property) making it efficient.
    /// 
    pub fn esdirk33() -> Self {
        // Parameters (computed in f64 for precision)
        let sqrt3 = 3.0_f64.sqrt();
        let beta = sqrt3 / 6.0 + 0.5;
        let gamma = (-1.0 / 8.0) * (sqrt3 + 1.0);

        let c = [0.0, 1.0, 0.5];
        let a = [
            [0.0, 0.0, 0.0],
            [4.0 * gamma + 2.0 * beta, 1.0 - 4.0 * gamma - 2.0 * beta, 0.0],
            [0.5 - beta - gamma, gamma, beta]
        ];
        let b = [1.0/6.0, 1.0/6.0, 2.0/3.0];

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
    /// ESDIRK3(2)4L[2]SA: 4-stage, 3rd order ESDIRK method with 2nd order embedding
    /// 
    /// This method is a high-quality 3rd order ESDIRK method with an embedded 2nd order method
    /// for adaptive step size control. It is L-stable and suitable for stiff problems.
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
        let d2_term1 = c3 * (-1.0 + 6.0 * g - 24.0 * g3 + 12.0 * g4 - 6.0 * g5) / 
                      (4.0 * g * (2.0 * g - c3) * (1.0 - 6.0 * g + 6.0 * g2));
        let d2_term2 = (3.0 - 27.0 * g + 68.0 * g2 - 55.0 * g3 + 21.0 * g4 - 6.0 * g5) /
                      (2.0 * (2.0 * g - c3) * (1.0 - 6.0 * g + 6.0 * g2));
        let d2 = d2_term1 + d2_term2;
        
        let d3 = -g * (-2.0 + 21.0 * g - 68.0 * g2 + 79.0 * g3 - 33.0 * g4 + 12.0 * g5) /
                 (c3 * (c3 - 2.0 * g) * (1.0 - 6.0 * g + 6.0 * g2));
        
        let d4 = -3.0 * g2 * (-1.0 + 4.0 * g - 2.0 * g2 + g3) / (1.0 - 6.0 * g + 6.0 * g2);
        let d1 = 1.0 - d2 - d3 - d4;

        let c = [0.0, 2.0 * g, 3.0 / 5.0, 1.0];
        let a = [
            [0.0, 0.0, 0.0, 0.0],
            [g, g, 0.0, 0.0],
            [a31, a32, g, 0.0],
            [b1, b2, b3, g]
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