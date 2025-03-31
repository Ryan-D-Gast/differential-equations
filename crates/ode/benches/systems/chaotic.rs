use super::*;

pub struct Lorenz {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl ODE<f64, 3, 1> for Lorenz {
    fn diff(&self, _t: f64, y: &SVector<f64, 3>, dydt: &mut SVector<f64, 3>) {
        dydt[0] = self.sigma * (y[1] - y[0]);
        dydt[1] = y[0] * (self.rho - y[2]) - y[1];
        dydt[2] = y[0] * y[1] - self.beta * y[2];
    }
}