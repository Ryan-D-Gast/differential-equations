// Defines systems for testing the ODE solvers

use differential_equations::ode::ODE;
use nalgebra::SVector;

/// Exponential growth, a simple first-order ODE
/// dy/dt = k * y
/// where k is a constant
pub struct ExponentialGrowth {
    pub k: f64,
}

impl ODE<f64, SVector<f64, 1>> for ExponentialGrowth {
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

impl ODE<f64, SVector<f64, 1>> for LinearEquation {
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

impl ODE<f64, SVector<f64, 2>> for HarmonicOscillator {
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

impl ODE<f64, SVector<f64, 1>> for LogisticEquation {
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

impl ODE<f64, SVector<f64, 2>> for VanDerPolOscillator {
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

impl ODE<f64, SVector<f64, 3>> for LorenzSystem {
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

impl ODE<f64, SVector<f64, 2>> for BrusselatorSystem {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        let y1 = y[0];
        let y2 = y[1];

        dydt[0] = self.a + y1 * y1 * y2 - (self.b + 1.0) * y1;
        dydt[1] = self.b * y1 - y1 * y1 * y2;
    }
}

/// Circular Restricted Three Body Problem (CR3BP)
pub struct Cr3bp {
    pub mu: f64, // CR3BP mass ratio
}

impl ODE<f64, SVector<f64, 6>> for Cr3bp {
    /// Differential equation for the initial value Circular Restricted Three
    /// Body Problem (CR3BP).
    /// All parameters are in non-dimensional form.
    fn diff(&self, _t: f64, y: &SVector<f64, 6>, dydt: &mut SVector<f64, 6>) {
        // Mass ratio
        let mu = self.mu;

        // Extracting states
        let (rx, ry, rz, vx, vy, vz) = (y[0], y[1], y[2], y[3], y[4], y[5]);

        // Distance to primary body
        let r13 = ((rx + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();
        // Distance to secondary body
        let r23 = ((rx - 1.0 + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();

        // Computing three-body dynamics
        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = vz;
        dydt[3] = rx + 2.0 * vy
            - (1.0 - mu) * (rx + mu) / r13.powi(3)
            - mu * (rx - 1.0 + mu) / r23.powi(3);
        dydt[4] = ry - 2.0 * vx - (1.0 - mu) * ry / r13.powi(3) - mu * ry / r23.powi(3);
        dydt[5] = -(1.0 - mu) * rz / r13.powi(3) - mu * rz / r23.powi(3);
    }
}
