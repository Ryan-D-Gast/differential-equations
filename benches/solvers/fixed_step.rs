//! Fixed Step NumericalMethod Benchmarks

use super::*;

macro_rules! bench_fixed_step {
    ($name:ident, $solver_fn:expr, $solver_name:expr, $system:expr, $y0:expr, $t0:expr, $t1:expr, $dt:expr) => {
        pub fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($solver_name);
            group.sample_size(10);
            group.bench_with_input(
                BenchmarkId::new(stringify!($system), "default"),
                &(),
                |b, _| {
                    b.iter(|| {
                        let mut solver = $solver_fn;
                        let problem = ODEProblem::new($system, $t0, $t1, $y0.clone());
                        black_box(problem.solve(&mut solver).unwrap());
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
    ExplicitRungeKutta::euler(0.1),
    "Euler",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_midpoint_ho,
    ExplicitRungeKutta::midpoint(0.1),
    "Midpoint",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_heun_ho,
    ExplicitRungeKutta::heun(0.1),
    "Heun",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_ralston_ho,
    ExplicitRungeKutta::ralston(0.1),
    "Ralston",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_rk4_ho,
    ExplicitRungeKutta::rk4(0.1),
    "RK4",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_three_eights_ho,
    ExplicitRungeKutta::three_eighths(0.1),
    "ThreeEights",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_apcf4_ho,
    AdamsPredictorCorrector::f4(0.1),
    "APCF4",
    HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1
);

// Van der Pol benchmarks
bench_fixed_step!(
    bench_euler_vdp,
    ExplicitRungeKutta::euler(0.01),
    "Euler",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_midpoint_vdp,
    ExplicitRungeKutta::midpoint(0.01),
    "Midpoint",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_heun_vdp,
    ExplicitRungeKutta::heun(0.01),
    "Heun",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_ralston_vdp,
    ExplicitRungeKutta::ralston(0.01),
    "Ralston",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_rk4_vdp,
    ExplicitRungeKutta::rk4(0.01),
    "RK4",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_three_eights_vdp,
    ExplicitRungeKutta::three_eighths(0.01),
    "ThreeEights",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);
bench_fixed_step!(
    bench_apcf4_vdp,
    AdamsPredictorCorrector::f4(0.01),
    "APCF4",
    VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01
);

// Lorenz system benchmarks (chaotic)
bench_fixed_step!(
    bench_euler_lorenz,
    ExplicitRungeKutta::euler(0.001),
    "Euler",
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
    ExplicitRungeKutta::midpoint(0.001),
    "Midpoint",
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
    ExplicitRungeKutta::heun(0.001),
    "Heun",
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
    ExplicitRungeKutta::ralston(0.001),
    "Ralston",
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
    ExplicitRungeKutta::rk4(0.01),
    "RK4",
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
    ExplicitRungeKutta::three_eighths(0.01),
    "ThreeEights",
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
    AdamsPredictorCorrector::f4(0.01),
    "APCF4",
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
    ExplicitRungeKutta::euler(0.1),
    "Euler",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_midpoint_exp,
    ExplicitRungeKutta::midpoint(0.1),
    "Midpoint",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_heun_exp,
    ExplicitRungeKutta::heun(0.1),
    "Heun",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_ralston_exp,
    ExplicitRungeKutta::ralston(0.1),
    "Ralston",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_rk4_exp,
    ExplicitRungeKutta::rk4(0.1),
    "RK4",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_three_eights_exp,
    ExplicitRungeKutta::three_eighths(0.1),
    "ThreeEights",
    Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1
);
bench_fixed_step!(
    bench_apcf4_exp,
    AdamsPredictorCorrector::f4(0.1),
    "APCF4",
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
    bench_rk4_exp,
    bench_three_eights_exp,
    bench_apcf4_exp,
);
