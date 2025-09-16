//! Adaptive Step NumericalMethod Benchmarks

use super::*;

/// Adaptive Step NumericalMethod Benchmarking Macro
macro_rules! bench_adaptive_step {
    ($name:ident, $solver_fn:expr, $solver_name:expr, $system:expr, $y0:expr, $t0:expr, $t1:expr, $h0:expr, $rtol:expr, $atol:expr) => {
        pub fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($solver_name);
            group.sample_size(10);
            group.bench_with_input(
                BenchmarkId::new(stringify!($system), "default"),
                &(),
                |b, _| {
                    b.iter(|| {
                        let mut solver = $solver_fn.rtol($rtol).atol($atol);
                        let problem = ODEProblem::new($system, $t0, $t1, $y0.clone());
                        black_box(problem.solve(&mut solver).unwrap());
                    });
                },
            );
            group.finish();
        }
    };
}

/// Dormand-Prince NumericalMethod Benchmarking Macro - compatible with DOP853 and DOPRI5
macro_rules! bench_dormand_prince {
    ($name:ident, $solver_fn:expr, $solver_name:expr, $system:expr, $y0:expr, $t0:expr, $t1:expr, $h0:expr, $rtol:expr, $atol:expr) => {
        pub fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($solver_name);
            group.sample_size(10);
            group.bench_with_input(
                BenchmarkId::new(stringify!($system), "default"),
                &(),
                |b, _| {
                    b.iter(|| {
                        let mut solver = $solver_fn.h0($h0).rtol($rtol).atol($atol);
                        let problem = ODEProblem::new($system, $t0, $t1, $y0.clone());
                        black_box(problem.solve(&mut solver).unwrap());
                    });
                },
            );
            group.finish();
        }
    };
}

// Benchmark for Harmonic Oscillator with all solvers
bench_adaptive_step!(
    bench_rkf_ho,
    ExplicitRungeKutta::rkf45(),
    "RKF45",
    &HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_adaptive_step!(
    bench_cashkarp_ho,
    ExplicitRungeKutta::cash_karp(),
    "CashKarp",
    &HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dopri5_ho,
    ExplicitRungeKutta::dopri5(),
    "DOPRI5",
    &HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dop853_ho,
    ExplicitRungeKutta::dop853(),
    "DOP853",
    &HarmonicOscillator,
    vector![1.0, 0.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);

// Benchmark for Van der Pol with all solvers
bench_adaptive_step!(
    bench_rkf_vdp,
    ExplicitRungeKutta::rkf45(),
    "RKF45",
    &VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01,
    1e-6,
    1e-6
);
bench_adaptive_step!(
    bench_cashkarp_vdp,
    ExplicitRungeKutta::cash_karp(),
    "CashKarp",
    &VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dopri5_vdp,
    ExplicitRungeKutta::dopri5(),
    "DOPRI5",
    &VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dop853_vdp,
    ExplicitRungeKutta::dop853(),
    "DOP853",
    &VanDerPol { mu: 1.0 },
    vector![2.0, 0.0],
    0.0,
    10.0,
    0.01,
    1e-6,
    1e-6
);

// Benchmark for Lorenz system with all solvers
bench_adaptive_step!(
    bench_rkf_lorenz,
    ExplicitRungeKutta::rkf45(),
    "RKF45",
    &Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001,
    1e-6,
    1e-6
);
bench_adaptive_step!(
    bench_cashkarp_lorenz,
    ExplicitRungeKutta::cash_karp(),
    "CashKarp",
    &Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dopri5_lorenz,
    ExplicitRungeKutta::dopri5(),
    "DOPRI5",
    &Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dop853_lorenz,
    ExplicitRungeKutta::dop853(),
    "DOP853",
    &Lorenz {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0
    },
    vector![1.0, 1.0, 1.0],
    0.0,
    10.0,
    0.001,
    1e-6,
    1e-6
);

// Benchmark for Exponential system with all solvers
bench_adaptive_step!(
    bench_rkf_exp,
    ExplicitRungeKutta::rkf45(),
    "RKF45",
    &Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_adaptive_step!(
    bench_cashkarp_exp,
    ExplicitRungeKutta::cash_karp(),
    "CashKarp",
    &Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dopri5_exp,
    ExplicitRungeKutta::dopri5(),
    "DOPRI5",
    &Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);
bench_dormand_prince!(
    bench_dop853_exp,
    ExplicitRungeKutta::dop853(),
    "DOP853",
    &Exponential { lambda: -0.5 },
    vector![1.0],
    0.0,
    10.0,
    0.1,
    1e-6,
    1e-6
);

criterion_group!(
    adaptive_step_benchmarks,
    // Harmonic Oscillator benchmarks
    bench_dopri5_ho,
    bench_rkf_ho,
    bench_cashkarp_ho,
    bench_dop853_ho,
    // Van der Pol benchmarks
    bench_dopri5_vdp,
    bench_rkf_vdp,
    bench_cashkarp_vdp,
    bench_dop853_vdp,
    // Lorenz system benchmarks
    bench_dopri5_lorenz,
    bench_rkf_lorenz,
    bench_cashkarp_lorenz,
    bench_dop853_lorenz,
    // Exponential system benchmarks
    bench_dopri5_exp,
    bench_rkf_exp,
    bench_cashkarp_exp,
    bench_dop853_exp
);
