//! Example 04: SIR Epidemiological Model
//!
//! This example models the spread of an infectious disease using the SIR model with ODEs:
//! dS/dt = -β * S * I / N
//! dI/dt = β * S * I / N - γ * I
//! dR/dt = γ * I
//!
//! where:
//! - S is the susceptible population
//! - I is the infected population
//! - R is the recovered population
//! - β is the transmission rate
//! - γ is the recovery rate
//! - N is the total population
//!
//! The SIR model is a foundational compartmental model in epidemiology used to
//! understand disease spread, predict outbreaks, and evaluate intervention strategies.
//!
//! This example demonstrates:
//! - Using custom structs with the #[derive(State)] attribute
//! - Implementing custom event detection with a custom enum type
//! - Setting up even-interval output points
//! - Working with solution status information

use differential_equations::prelude::*;
use quill::*;

/// SIR (Susceptible, Infected, Recovered) Model
struct SIRModel {
    beta: f64,       // Transmission rate
    gamma: f64,      // Recovery rate
    population: f64, // Total population
}

// There are two major difference between this example and the previous one:
// 1. Instead of having the ControlFlag contain a string, we have a custom enum PopulationMonitor that contains the reason for termination.
// 2. Instead of a float type, or nalgabra matrix/vector type, we use the derive state macro to create our own struct
//    that contains the state variables of the SIR model. This struct implements the State trait, which allows us to use it with the ODE solver.
impl ODE<f64, SIRState<f64>, PopulationMonitor> for SIRModel {
    fn diff(&self, _t: f64, y: &SIRState<f64>, dydt: &mut SIRState<f64>) {
        let s = y.susceptible; // Susceptible
        let i = y.infected; // Infected
        let _r = y.recovered; // Recovered

        dydt.susceptible = -self.beta * s * i / self.population; // Susceptible
        dydt.infected = self.beta * s * i / self.population - self.gamma * i; // Infected
        dydt.recovered = self.gamma * i; // Recovered
    }

    fn event(&self, _t: f64, y: &SIRState<f64>) -> ControlFlag<f64, SIRState<f64>, PopulationMonitor> {
        let i = y.infected; // Infected

        // Check the PopulationMonitor

        // Terminate the simulation when the number of infected individuals falls below 1
        if i < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::InfectedBelowOne)
        } else if y.population() < 1.0 {
            ControlFlag::Terminate(PopulationMonitor::PopulationDiedOut)
        } else {
            ControlFlag::Continue
        }
    }
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

#[derive(State)]
struct SIRState<T> {
    susceptible: T,
    infected: T,
    recovered: T,
}

impl SIRState<f64> {
    /// Returns the total population.
    fn population(&self) -> f64 {
        self.susceptible + self.infected + self.recovered
    }
}

fn main() {
    // method with relative and absolute tolerances
    let mut method = AdamsPredictorCorrector::v4().tol(1e-6);

    // Initial State
    let y0 = SIRState {
        susceptible: 990.0,
        infected: 10.0,
        recovered: 0.0,
    };
    let population = y0.population(); // Total population

    // Simulation time
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
    let sir_problem = ODEProblem::new(ode, t0, tf, y0);

    // Solve the ode with even output at interval dt: 1.0
    match sir_problem
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
                    t, y.susceptible, y.infected, y.recovered
                );
            }

            // Print the statistics
            println!("Function evaluations: {}", solution.evals.function);
            println!("Steps: {}", solution.steps.total());
            println!("Rejected Steps: {}", solution.steps.rejected);
            println!("Accepted Steps: {}", solution.steps.accepted);
            println!("Solve time: {:?} seconds", solution.timer.elapsed());

            // Plot the solution using quill
            Plot::builder()
                .title("SIR Epidemiological Model")
                .x_label("Time (days)")
                .y_label("Population")
                .legend(Legend::TopRightInside)
                .data([
                    Series::builder()
                        .name("Susceptible")
                        .color("Blue")
                        .data(
                            solution
                                .iter()
                                .map(|(t, y)| (*t, y.susceptible))
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                    Series::builder()
                        .name("Infected")
                        .color("Red")
                        .data(
                            solution
                                .iter()
                                .map(|(t, y)| (*t, y.infected))
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                    Series::builder()
                        .name("Recovered")
                        .color("Green")
                        .data(
                            solution
                                .iter()
                                .map(|(t, y)| (*t, y.recovered))
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                ])
                .build()
                .to_svg("examples/ode/04_sir_model/sir_model.svg")
                .expect("Failed to save plot as SVG");
        }
        Err(e) => panic!("Error: {:?}", e),
    };
}
