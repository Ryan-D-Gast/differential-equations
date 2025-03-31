use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

// Global storage for solution data
static SOLUTION_DATA: LazyLock<Mutex<HashMap<String, SolutionData>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct SolutionData {
    pub solver: String,
    pub system: String,
    pub final_t: f64,
    pub final_y: Vec<f64>,
    pub time_ns: u64,
}

pub fn record_solution(solver: &str, system: &str, final_t: f64, final_y: &[f64], time_ns: u64) {
    let mut data = SOLUTION_DATA.lock().unwrap();
    let key = format!("{}_{}", solver, system);
    
    data.insert(key, SolutionData {
        solver: solver.to_string(),
        system: system.to_string(),
        final_t,
        final_y: final_y.to_vec(),
        time_ns,
    });
}

pub fn generate_report() {
    let data = SOLUTION_DATA.lock().unwrap();
    
    // Group by system
    let mut by_system: HashMap<String, Vec<&SolutionData>> = HashMap::new();
    
    for (_, solution) in data.iter() {
        by_system
            .entry(solution.system.clone())
            .or_default()
            .push(solution);
    }
    
    println!("--- Solution Comparison Report ---");
    for (system, solutions) in &by_system {
        println!("\nODE: {}", system);
        println!("{:<15} {:<10} {:<20} {:<20}", "Solver", "Time (t)", "Final State (y)", "Runtime (ms)");
        
        for sol in solutions {
            println!("{:<15} {:<10.6} {:<20} {:<20.3}", 
                sol.solver, 
                sol.final_t, 
                format!("{:?}", sol.final_y), 
                sol.time_ns as f64 / 1_000_000.0);
        }
    }
}