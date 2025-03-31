use super::*;

pub struct HarmonicOscillator;

impl ODE<f64, 2, 1> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

pub struct VanDerPol {
    pub mu: f64,
}

impl ODE<f64, 2, 1> for VanDerPol {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
}