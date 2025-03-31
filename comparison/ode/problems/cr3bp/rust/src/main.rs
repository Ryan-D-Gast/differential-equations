use differential_equations::ode::*;
use nalgebra::{SVector, vector};
use std::env;

pub struct Cr3bp {
    pub mu: f64,
}

impl ODE<f64, 6> for Cr3bp {
    fn diff(&self, _t: f64, y: &SVector<f64, 6>, dydt: &mut SVector<f64, 6>) {
        let mu = self.mu;

        let (rx, ry, rz, vx, vy, vz) = (y[0], y[1], y[2], y[3], y[4], y[5]);

        let r13 = ((rx + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();
        let r23 = ((rx - 1.0 + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();

        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = vz;
        dydt[3] = rx + 2.0 * vy - (1.0 - mu) * (rx + mu) / r13.powi(3) - mu * (rx - 1.0 + mu) / r23.powi(3);
        dydt[4] = ry - 2.0 * vx - (1.0 - mu) * ry / r13.powi(3) - mu * ry / r23.powi(3);
        dydt[5] = -(1.0 - mu) * rz / r13.powi(3) - mu * rz / r23.powi(3);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 12 {
        eprintln!("Usage: {} <mu> <t0> <tf> <y0_0> <y0_1> <y0_2> <y0_3> <y0_4> <y0_5> <atol> <rtol>", args[0]);
        std::process::exit(1);
    }

    let csv = args.iter().any(|arg| arg == "--csv");

    let mu: f64 = args[1].parse().unwrap();
    let t0: f64 = args[2].parse().unwrap();
    let tf: f64 = args[3].parse().unwrap();
    let y0: SVector<f64, 6> = vector![args[4].parse().unwrap(), args[5].parse().unwrap(), args[6].parse().unwrap(), args[7].parse().unwrap(), args[8].parse().unwrap(), args[9].parse().unwrap()];
    let atol: f64 = args[10].parse().unwrap();
    let rtol: f64 = args[11].parse().unwrap();

    let system = Cr3bp { mu };
    let mut solver = DOP853::new().rtol(rtol).atol(atol);
    let ivp = IVP::new(system, t0, tf, y0);
    
    let start = std::time::Instant::now();
    let result = ivp.solve(&mut solver).unwrap();
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    if csv {
        result.to_csv("../target/rust_cr3bp.csv").unwrap();
    }

    let (_t, y) = result.last().unwrap();
    
    println!("[{:.15e}, {:.15e}, {:.15e}, {:.15e}, {:.15e}, {:.15e}] {:.6}", y[0], y[1], y[2], y[3], y[4], y[5], elapsed_secs);
}