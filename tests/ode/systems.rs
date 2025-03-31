// Defines systems for testing the ODE solvers

use nalgebra::SVector;
use ode::ODE;

pub struct ExponentialGrowth {
    pub k: f64,
}

impl ODE<f64, 1, 1> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0];
    }
}

pub struct LinearEquation {
    pub a: f64,
    pub b: f64,
}

impl ODE<f64, 1, 1> for LinearEquation {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.a + self.b * y[0];
    }
}

pub struct HarmonicOscillator {
    pub k: f64,
}

impl ODE<f64, 2, 1> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.k * y[0];
    }
}

pub struct LogisticEquation {
    pub k: f64,
    pub m: f64,
}

impl ODE<f64, 1, 1> for LogisticEquation {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0] * (1.0 - y[0] / self.m);
    }
}
