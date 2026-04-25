sed -i 's/\.solout(solout)/.solout(solout.clone())/g' examples/ode/10_custom_solout/main.rs
sed -i '45i\#[derive(Clone)]' examples/ode/10_custom_solout/main.rs
