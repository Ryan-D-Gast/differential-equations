use ode::*;
use nalgebra::{SVector, vector};
use std::env;

// Define the van der Pol oscillator system
struct VanDerPol {
    pub mu: f64,
}

impl ODE<f64, 2> for VanDerPol {
    fn diff(&self, _t: f64, y: &SVector<f64, 2>, dydt: &mut SVector<f64, 2>) {
        dydt[0] = y[1];
        dydt[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!("Usage: {} <mu> <t0> <tf> <y0_0> <y0_1> <atol> <rtol>", args[0]);
        std::process::exit(1);
    }

    // Check for CSV flag
    let csv = args.iter().any(|arg| arg == "--csv");

    let mu: f64 = args[1].parse().unwrap();
    let t0: f64 = args[2].parse().unwrap();
    let tf: f64 = args[3].parse().unwrap();
    let y0: SVector<f64, 2> = vector![args[4].parse().unwrap(), args[5].parse().unwrap()];
    let atol: f64 = args[6].parse().unwrap();
    let rtol: f64 = args[7].parse().unwrap();

    let system = VanDerPol { mu };
    let mut solver = DOP853::new().rtol(rtol).atol(atol);
    let ivp = IVP::new(system, t0, tf, y0);
    
    let start = std::time::Instant::now();
    let result = ivp.solve(&mut solver).unwrap();
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    if csv {
        result.to_csv("../target/rust_vanderpol.csv").unwrap();
    }

    let (_t, y) = result.last().unwrap();
    
    // Print in parsable format: [y0, y1] time
    println!("[{:.15e}, {:.15e}] {:.6}", y[0], y[1], elapsed_secs);
}