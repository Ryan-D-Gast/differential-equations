use super::*;

pub struct Exponential {
    pub lambda: f64,
}

impl ODE<f64, SVector<f64, 1>> for Exponential {
    fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
        dydt[0] = self.lambda * y[0];
    }
}
