use criterion::criterion_main;

mod solvers;
mod systems;

criterion_main! {
    solvers::fixed_step::fixed_step_benchmarks,
    solvers::adaptive_step::adaptive_step_benchmarks,
}
