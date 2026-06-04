use differential_equations::methods::Milstein;
use differential_equations::prelude::*;

struct MultiDimensionalNoiseTest {
    // We will use a fixed sequence for testing multidimensional correctness
    dw_sequence: Vec<[f64; 2]>,
    idx: usize,
}

impl MultiDimensionalNoiseTest {
    fn new(dw_sequence: Vec<[f64; 2]>) -> Self {
        Self {
            dw_sequence,
            idx: 0,
        }
    }
}

impl SDE<f64, [f64; 2]> for MultiDimensionalNoiseTest {
    fn drift(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = -y[0];
        dydt[1] = -2.0 * y[1];
    }

    fn diffusion(&self, _t: f64, y: &[f64; 2], dydw: &mut [f64; 2]) {
        dydw[0] = 0.5 * y[0];
        dydw[1] = 0.1 * y[1];
    }

    fn noise(&mut self, _dt: f64, dw: &mut [f64; 2]) {
        if self.idx < self.dw_sequence.len() {
            dw[0] = self.dw_sequence[self.idx][0];
            dw[1] = self.dw_sequence[self.idx][1];
            self.idx += 1;
        } else {
            dw[0] = 0.0;
            dw[1] = 0.0;
        }
    }
}

#[test]
fn test_multidimensional_milstein() {
    let t0 = 0.0;
    let tf = 1.0;
    let y0 = [1.0, 1.0];

    // Create a small deterministic dw sequence for reproducibility
    let dw_sequence = vec![[0.1, -0.1], [-0.05, 0.2], [0.15, -0.05], [0.0, 0.1]];
    let steps = dw_sequence.len();
    let h = (tf - t0) / (steps as f64);

    let mut sde_milstein = MultiDimensionalNoiseTest::new(dw_sequence);
    let sol = IVP::sde(&mut sde_milstein, t0, tf, y0)
        .method(Milstein::new(h))
        .solve()
        .unwrap();

    assert_eq!(sol.t.len(), steps + 1);

    // Verify that integration happened and states changed
    let final_y = sol.y.last().unwrap();
    assert!(final_y[0] != 1.0);
    assert!(final_y[1] != 1.0);
}
