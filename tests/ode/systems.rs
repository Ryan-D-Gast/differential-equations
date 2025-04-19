// Defines systems for testing the ODE solvers

use differential_equations::ode::ODE;
use nalgebra::SVector;

/// Exponential growth, a simple first-order ODE
/// dy/dt = k * y
/// where k is a constant
pub struct ExponentialGrowth {
    pub k: f64,
}

impl ODE<f64, 1, 1> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0];
    }
}

/// Linear equation, a simple first-order linear ODE
/// dy/dt = a + b * y
/// where a and b are constants
pub struct LinearEquation {
    pub a: f64,
    pub b: f64,
}

impl ODE<f64, 1, 1> for LinearEquation {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.a + self.b * y[0];
    }
}

/// Harmonic oscillator, a simple mechanical system
/// dx/dt = v
/// dv/dt = -k * x
/// where k is the spring constant
pub struct HarmonicOscillator {
    pub k: f64,
}

impl ODE<f64, 2, 1> for HarmonicOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.k * y[0];
    }
}

/// Logistic equation, a model for population growth
/// dy/dt = k * y * (1 - y/m)
/// where k is the growth rate and m is the carrying capacity
pub struct LogisticEquation {
    pub k: f64,
    pub m: f64,
}

impl ODE<f64, 1, 1> for LogisticEquation {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.k * y[0] * (1.0 - y[0] / self.m);
    }
}

/// Van der Pol oscillator, a nonlinear oscillator
/// dx/dt = y
/// dy/dt = mu * (1 - x^2) * y - x
/// where mu is a parameter that controls the nonlinearity
pub struct VanDerPolOscillator {
    pub mu: f64,
}

impl ODE<f64, 2, 1> for VanDerPolOscillator {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        let y1 = y[0];
        let y2 = y[1];
        
        dydt[0] = y2;
        dydt[1] = self.mu * (1.0 - y1 * y1) * y2 - y1;
    }
}

/// Lorenz system, a set of chaotic equations
/// dx/dt = sigma * (y - x)
/// dy/dt = x * (rho - z) - y
/// dz/dt = x * y - beta * z
/// where sigma, rho, and beta are constants
pub struct LorenzSystem {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl ODE<f64, 3, 1> for LorenzSystem {
    fn diff(&self, _t: f64, y: &SVector<f64, 3>, dydt: &mut SVector<f64, 3>) {
        let x = y[0];
        let y_val = y[1];
        let z = y[2];
        
        dydt[0] = self.sigma * (y_val - x);
        dydt[1] = x * (self.rho - z) - y_val;
        dydt[2] = x * y_val - self.beta * z;
    }
}

/// Brusselator system, an autocatalytic chemical reaction model
/// dy1/dt = a + y1^2 * y2 - (b + 1) * y1
/// dy2/dt = b * y1 - y1^2 * y2
/// where a and b are constants
pub struct BrusselatorSystem {
    pub a: f64,
    pub b: f64,
}

impl ODE<f64, 2, 1> for BrusselatorSystem {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        let y1 = y[0];
        let y2 = y[1];
        
        dydt[0] = self.a + y1 * y1 * y2 - (self.b + 1.0) * y1;
        dydt[1] = self.b * y1 - y1 * y1 * y2;
    }
}