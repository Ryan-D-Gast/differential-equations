//! Defines systems to test the DDE solvers

use differential_equations::dde::DDE;
use nalgebra::SVector;

#[derive(Clone)]
pub struct MackeyGlass {
    pub beta: f64,
    pub gamma: f64,
    pub n: f64,
    pub tau: f64,
}

impl DDE<1, f64, SVector<f64, 1>> for MackeyGlass {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, yd: &[SVector<f64, 1>; 1], dydt: &mut SVector<f64, 1>) {
        dydt[0] = (self.beta * yd[0][0]) / (1.0 + yd[0][0].powf(self.n)) - self.gamma * y[0];
    }

    fn lags(&self, _t: f64, _y: &SVector<f64, 1>, lags: &mut [f64; 1]) {
        lags[0] = self.tau;
    }
}

#[derive(Clone)]
pub struct ExponentialGrowth {
    pub k: f64,
}

impl DDE<0, f64, SVector<f64, 1>> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, _yd: &[SVector<f64, 1>; 0], dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0];
    }

    fn lags(&self, _t: f64, _y: &SVector<f64, 1>, _lags: &mut [f64; 0]) {
        // No lags for this system just using to test interpolation
    }
}
