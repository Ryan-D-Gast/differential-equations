// Defines systems to test the DDE solvers

use differential_equations::dde::DDE;

#[derive(Clone)]
pub struct MackeyGlass {
    pub beta: f64,
    pub gamma: f64,
    pub n: f64,
    pub tau: f64,
}

impl DDE<1, f64, f64> for MackeyGlass {
    fn diff(&self, _t: f64, y: &f64, yd: &[f64; 1], dydt: &mut f64) {
        *dydt = (self.beta * yd[0]) / (1.0 + yd[0].powf(self.n)) - self.gamma * *y;
    }

    fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
        lags[0] = self.tau;
    }
}

#[derive(Clone)]
pub struct ExponentialGrowth {
    pub k: f64,
}

impl DDE<1, f64, f64> for ExponentialGrowth {
    fn diff(&self, _t: f64, y: &f64, _yd: &[f64; 1], dydt: &mut f64) {
        *dydt = self.k * *y;
    }

    fn lags(&self, _t: f64, _y: &f64, lags: &mut [f64; 1]) {
        lags[0] = 0.0; // Dummy lag because L > 0
    }
}
