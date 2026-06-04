use differential_equations::{bvp::Boundary, ode::ODE};

pub trait OdeBoundary: ODE<f64, [f64; 2]> + Boundary<f64, [f64; 2]> {}

impl<T> OdeBoundary for T where T: ODE<f64, [f64; 2]> + Boundary<f64, [f64; 2]> {}

pub struct HarmonicOscillatorBvp {
    pub target: f64,
}

impl ODE<f64, [f64; 2]> for HarmonicOscillatorBvp {
    fn diff(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

impl Boundary<f64, [f64; 2]> for HarmonicOscillatorBvp {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        res[0] = y_a[0];
        res[1] = y_b[0] - self.target;
    }
}

pub struct PipeHeatTransfer {
    pub ambient_temperature: f64,
    pub heat_loss_rate: f64,
    pub inlet_temperature: f64,
}

impl PipeHeatTransfer {
    pub fn analytical_initial_gradient(&self, length: f64) -> f64 {
        let theta_0 = self.inlet_temperature - self.ambient_temperature;
        let beta_l = self.heat_loss_rate * length;
        -self.heat_loss_rate * theta_0 * beta_l.tanh()
    }

    pub fn analytical_outlet_temperature(&self, length: f64) -> f64 {
        let theta_0 = self.inlet_temperature - self.ambient_temperature;
        let beta_l = self.heat_loss_rate * length;
        self.ambient_temperature + theta_0 / beta_l.cosh()
    }
}

impl ODE<f64, [f64; 2]> for PipeHeatTransfer {
    fn diff(&self, _x: f64, y: &[f64; 2], dydx: &mut [f64; 2]) {
        dydx[0] = y[1];
        dydx[1] = self.heat_loss_rate.powi(2) * (y[0] - self.ambient_temperature);
    }
}

impl Boundary<f64, [f64; 2]> for PipeHeatTransfer {
    fn boundary(&self, y_a: &[f64; 2], y_b: &[f64; 2], res: &mut [f64; 2]) {
        res[0] = y_a[0] - self.inlet_temperature;
        res[1] = y_b[1];
    }
}
