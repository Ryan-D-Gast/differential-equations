sed -i 's/let sol = problem.clone().method(tmp_solver).solve().unwrap();/let sol = problem.clone().method(tmp_solver.clone()).solve().unwrap();/g' tests/ode/comparison.rs
