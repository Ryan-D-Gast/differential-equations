use differential_equations::ode::*;
use nalgebra::{SVector, vector};

struct HarmonicOscillator {
    k: f32,
}

impl ODE<f32, 2> for HarmonicOscillator {
    fn diff(&self, _t: f32, y: &SVector<f32, 2>, dydt: &mut SVector<f32, 2>) {
        dydt[0] = y[1];
        dydt[1] = -self.k * y[0];
    }
}

fn main() {
    // Note how unlike 01_exponential_growth/main.rs, no intermediate variables are used and the IVP is setup and solved in one step.
    let solution = match IVP::new(
            HarmonicOscillator { k: 1.0 }, 
            0.0, 
            10.0,
            vector![1.0, 0.0]
        ).solve(&mut RK4::new(0.01)) {
        Ok(solution) => solution,
        Err(e) => panic!("Error: {:?}", e),
    };
    let (tf, yf) = solution.last().unwrap();
    println!("Solution: ({:?}, {:?})", tf, yf);
}
