find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(solver.clone())/.method(solver)/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(reference_solver.clone())/.method(reference_solver)/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(var_solver.clone())/.method(var_solver)/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(solver_dense.clone())/.method(solver_dense)/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(solver_even.clone())/.method(solver_even)/g' {} +
find examples tests -type f -name "*.rs" -exec sed -i -e 's/\.method(solver_t_out.clone())/.method(solver_t_out)/g' {} +
sed -i 's/match Ivp::ode(\&HarmonicOscillator { k: 1.0 }, 0.0, 10.0, vector!\[1.0, 0.0\])\n            .method(ExplicitRungeKutta::rk4(0.01_f64)).solve()/match Ivp::ode(\&HarmonicOscillator { k: 1.0 }, 0.0_f64, 10.0_f64, vector!\[1.0_f64, 0.0_f64\])\n            .method(ExplicitRungeKutta::rk4(0.01_f64)).solve()/g' examples/ode/02_harmonic_oscillator/main.rs
