//! Fixed Step NumericalMethod Benchmarks

use super::*;

macro_rules! bench_fixed_step {
    ($name:ident, $solver:ident, $system:expr, $y0:expr, $t0:expr, $t1:expr, $dt:expr) => {
        pub fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group(stringify!($solver));
            group.sample_size(10);
            group.bench_with_input(
                BenchmarkId::new(stringify!($system), "default"),
                &(),
                |b, _| {
                    b.iter(|| {
                        let mut solver = $solver::new($dt);
                        let ivp = IVP::new($system, $t0, $t1, $y0.clone());
                        black_box(ivp.solve(&mut solver).unwrap());
                    });
                },
            );
            group.finish();
        }
    };
}

// Harmonic Oscillator benchmarks
bench_fixed_step!(
    bench_euler_ho,
    Euler,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_midpoint_ho,
    Midpoint,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_heun_ho,
    Heun,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_ralston_ho,
    Ralston,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_rk4_ho,
    RK4,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_three_eights_ho,
    ThreeEights,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_apcf4_ho,
    APCF4,
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);

// Van der Pol benchmarks
bench_fixed_step!(
    bench_euler_vdp,
    Euler,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_midpoint_vdp,
    Midpoint,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_heun_vdp,
    Heun,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_ralston_vdp,
    Ralston,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_rk4_vdp,
    RK4,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_three_eights_vdp,
    ThreeEights,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_apcf4_vdp,
    APCF4,
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);

// Lorenz system benchmarks (chaotic)
bench_fixed_step!(
    bench_euler_lorenz,
    Euler,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001
);
bench_fixed_step!(
    bench_midpoint_lorenz,
    Midpoint,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001
);
bench_fixed_step!(
    bench_heun_lorenz,
    Heun,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001
);
bench_fixed_step!(
    bench_ralston_lorenz,
    Ralston,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001
);
bench_fixed_step!(
    bench_rk4_lorenz,
    RK4,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_three_eights_lorenz,
    ThreeEights,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_apcf4_lorenz,
    APCF4,
    Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.01
);

// Exponential system benchmarks (linear)
bench_fixed_step!(
    bench_euler_exp,
    Euler,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_midpoint_exp,
    Midpoint,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_heun_exp,
    Heun,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_ralston_exp,
    Ralston,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_rk4_exp,
    RK4,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_three_eights_exp,
    ThreeEights,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_apcf4_exp,
    APCF4,
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);

criterion_group!(
    fixed_step_benchmarks,
    // Harmonic Oscillator benchmarks
    bench_euler_ho,
    bench_midpoint_ho,
    bench_heun_ho,
    bench_ralston_ho,
    bench_rk4_ho,
    bench_three_eights_ho,
    bench_apcf4_ho,
    // Van der Pol benchmarks
    bench_euler_vdp,
    bench_midpoint_vdp,
    bench_heun_vdp,
    bench_ralston_vdp,
    bench_rk4_vdp,
    bench_three_eights_vdp,
    bench_apcf4_vdp,
    // Lorenz system benchmarks
    bench_euler_lorenz,
    bench_midpoint_lorenz,
    bench_heun_lorenz,
    bench_ralston_lorenz,
    bench_rk4_lorenz,
    bench_three_eights_lorenz,
    bench_apcf4_lorenz,
    // Exponential system benchmarks
    bench_euler_exp,
    bench_midpoint_exp,
    bench_heun_exp,
    bench_ralston_exp,
    bench_midpoint_exp,
    bench_three_eights_exp,
    bench_rk4_exp,
    bench_apcf4_exp,
);
