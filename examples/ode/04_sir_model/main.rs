use differential_equations::ode::methods::APCV4;
use differential_equations::ode::*;
use nalgebra::{vector, Vector3};

/// SIR (Susceptible, Infected, Recovered) Model
///
/// This struct defines the parameters for the SIR model.
///
struct SIRModel {
    beta: f64,       // Transmission rate
    gamma: f64,      // Recovery rate
    population: f64, // Total population
}

#[derive(Debug, Clone)]
enum PopulationMonitor {
    InfectedBelowOne,
    PopulationDiedOut,
}

impl std::fmt::Display for PopulationMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PopulationMonitor::InfectedBelowOne => write!(f, "Infected population is below 1"),
            PopulationMonitor::PopulationDiedOut => write!(f, "Population died out"),
        }
    }
}

// Unlike other examples where ODE defaults to have the final generic type as String.
// Here, the ODE trait is implemented with the final generic type as PopulationMonitor.
// Custom types that implement Clone and Debug can be used and passed through back
// to user when a termination event occurs.
impl ODE<f64, Vector3<f64>, PopulationMonitor> for SIRModel {
    fn diff(&self, _t: f64, y: &Vector3<f64>, dydt: &mut Vector3<f64>) {
        let s = y[0]; // Susceptible
        let i = y[1]; // Infected
        let _r = y[2]; // Recovered

        dydt[0] = -self.beta * s * i / self.population; // Susceptible
        dydt[1] = self.beta * s * i / self.population - self.gamma * i; // Infected
        dydt[2] = self.gamma * i; // Recovered
    }

    fn event(&self, _t: f64, y: &Vector3<f64>) -> ControlFlag<PopulationMonitor> {
        let i = y[1]; // Infected

        // Check the PopulationMonitor

        // Terminate the simulation when the number of infected individuals falls below 1
        if i < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::InfectedBelowOne)
        } else if y.iter().sum::<f64>() < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::PopulationDiedOut)
        } else {
            ControlFlag::Continue
        }
    }
}

fn main() {
    // method with relative and absolute tolerances
    let mut method = APCV4::new().tol(1e-6);

    // Initial conditions
    let initial_susceptible = 990.0;
    let initial_infected = 10.0;
    let initial_recovered = 0.0;
    let population = initial_susceptible + initial_infected + initial_recovered;

    let y0 = vector![initial_susceptible, initial_infected, initial_recovered];
    let t0 = 0.0;
    let tf = 100.0;

    // SIR model parameters
    let beta = 1.42; // Transmission rate
    let gamma = 0.14; // Recovery rate

    let ode = SIRModel {
        beta,
        gamma,
        population,
    };
    let sir_ivp = IVP::new(ode, t0, tf, y0);

    // Solve the ode with even output at interval dt: 1.0
    match sir_ivp
        .even(1.0)  // sets t-out at interval dt: 1.0
        .solve(&mut method) // Solve the ode and return the solution
    {
        Ok(solution) => {
            // Check if the solver stopped due to the event command
            if let Status::Interrupted(ref reason) = solution.status {
                // State the reason why the solver stopped
                println!("solver stopped: {}", reason);
            }

            // Print the solution
            println!("Solution:");
            println!("Time, Susceptible, Infected, Recovered");
            for (t, y) in solution.iter() {
                println!(
                    "{:.4}, {:.4}, {:.4}, {:.4}",
                    t, y[0], y[1], y[2]
                );
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals);
            println!("Steps: {}", solution.steps);
            println!("Rejected Steps: {}", solution.rejected_steps);
            println!("Accepted Steps: {}", solution.accepted_steps);
            println!("Solve time: {:?} seconds", solution.timer.elapsed());
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
