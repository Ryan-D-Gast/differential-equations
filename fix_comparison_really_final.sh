sed -i 's/let sol = problem.clone().method(solver.clone()).solve().unwrap();/let sol = problem.clone().method(solver).solve().unwrap();/g' tests/ode/comparison.rs
