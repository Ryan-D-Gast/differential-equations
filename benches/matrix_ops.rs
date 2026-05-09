use criterion::{Criterion, criterion_group};
use differential_equations::linalg::matrix::Matrix;

pub fn matrix_ops_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    let n = 1000;

    // Setup matrices
    let mut m1: Matrix<f64> = Matrix::full(n, n);
    let mut m2: Matrix<f64> = Matrix::full(n, n);
    for i in 0..n {
        for j in 0..n {
            m1[(i, j)] = (i + j) as f64;
            m2[(i, j)] = (i * j) as f64;
        }
    }

    group.bench_function("matrix_add_1000x1000", |b| {
        b.iter(|| {
            let m1_clone = m1.clone();
            let m2_clone = m2.clone();
            let _ = std::hint::black_box(m1_clone + m2_clone);
        })
    });

    group.bench_function("matrix_sub_1000x1000", |b| {
        b.iter(|| {
            let m1_clone = m1.clone();
            let m2_clone = m2.clone();
            let _ = std::hint::black_box(m1_clone - m2_clone);
        })
    });

    group.bench_function("matrix_component_mul_1000x1000", |b| {
        b.iter(|| {
            let m1_clone = m1.clone();
            let _ = std::hint::black_box(m1_clone.component_mul(2.0));
        })
    });

    let n_mul = 200;
    let mut mm1: Matrix<f64> = Matrix::full(n_mul, n_mul);
    let mut mm2: Matrix<f64> = Matrix::full(n_mul, n_mul);
    for i in 0..n_mul {
        for j in 0..n_mul {
            mm1[(i, j)] = (i + j) as f64;
            mm2[(i, j)] = (i * j) as f64;
        }
    }

    group.bench_function("matrix_mul_200x200", |b| {
        b.iter(|| {
            let m1_clone = mm1.clone();
            let m2_clone = mm2.clone();
            let _ = std::hint::black_box(m1_clone * m2_clone);
        })
    });

    group.finish();
}

criterion_group!(matrix_benchmarks, matrix_ops_benchmark);
