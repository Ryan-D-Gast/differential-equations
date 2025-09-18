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
use quill::prelude::*;

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
impl ODE<f64, SIRState<f64>> for SIRModel {
    fn diff(&self, _t: f64, y: &SIRState<f64>, dydt: &mut SIRState<f64>) {
        let s = y.susceptible;
        let i = y.infected;
        let _r = y.recovered;

        dydt.susceptible = -self.beta * s * i / self.population;
        dydt.infected = self.beta * s * i / self.population - self.gamma * i;
        dydt.recovered = self.gamma * i;
    }
}

impl Event<f64, SIRState<f64>> for SIRModel {
    fn config(&self) -> EventConfig {
        EventConfig::default().terminal() // Will terminate after the first event
    }

    /// Check when the number of infected individuals drops below 1 and if so, terminate the solver.
    fn event(&self, _t: f64, y: &SIRState<f64>) -> f64 {
        y.infected - 1.0
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
    // v4 refers to an adaptive step size 4th order Adams-Bashforth-Moulton method.
    let mut method = AdamsPredictorCorrector::v4().tol(1e-6);

    // Define the SIR model parameters and initial conditions
    let y0 = SIRState {
        susceptible: 990.0,
        infected: 10.0,
        recovered: 0.0,
    };
    let t0 = 0.0;
    let tf = 100.0;
    let population = y0.population();
    let beta = 1.42; // Transmission rate
    let gamma = 0.14; // Recovery rate
    let ode = SIRModel {
        beta,
        gamma,
        population,
    };
    let sir_problem = ODEProblem::new(&ode, t0, tf, y0);

    // Solve the SIR model problem with even output points every 1.0 time unit
    match sir_problem.even(1.0).event(&ode).solve(&mut method) {
        Ok(solution) => {
            // Check for event termination
            if let Status::Interrupted = solution.status {
                println!("solver stopped by event: Infected population dropped below 1");
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

            // Plotting
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
