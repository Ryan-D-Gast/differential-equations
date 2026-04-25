sed -i 's/let sol = problem.clone().method(solver).solve().unwrap();/let sol = problem.clone().method(solver.clone()).solve().unwrap();/g' tests/ode/comparison.rs
sed -i 's/let mut solver = $solver;/let mut solver = $solver.clone();/g' tests/ode/comparison.rs
