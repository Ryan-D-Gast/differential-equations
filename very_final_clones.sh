sed -i 's/struct TwoBodyODE/#[derive(Clone)]\nstruct TwoBodyODE/g' examples/ode/14_r2bp_stm/main.rs
sed -i 's/let sol = problem.method(solver).solve().unwrap();/let sol = problem.clone().method(solver.clone()).solve().unwrap();/g' examples/ode/14_r2bp_stm/main.rs
sed -i 's/let sol_flt = problem.method(solver).solve().unwrap();/let sol_flt = problem.clone().method(solver).solve().unwrap();/g' examples/ode/14_r2bp_stm/main.rs
sed -i 's/\.solve(&mut ExplicitRungeKutta::rk4(0.01))/.method(ExplicitRungeKutta::rk4(0.01)).solve()/g' examples/ode/02_harmonic_oscillator/main.rs
