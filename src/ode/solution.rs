//! Solution of IVP Problem returned by IVP::solve

use crate::ode::{SolverStatus, EventData};
use nalgebra::SMatrix;
use crate::traits::Real;

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Solution of IVP Problem returned by ODE Solvers
///
/// # Fields
/// * `y`              - Outputted dependent variable points.
/// * `t`              - Outputted independent variable points.
/// * `h`              - Predicted step size of last accepted step.
/// * `evals`          - Number of function evaluations.
/// * `steps`          - Number of steps.
/// * `rejected_steps` - Number of rejected steps.
/// * `accepted_steps` - Number of accepted steps.
///
#[derive(Debug, Clone)]
pub struct Solution<T, const R: usize, const C: usize, E> 
where 
    T: Real,
    E: EventData,
{
    /// Outputted independent variable points.
    pub t: Vec<T>,

    /// Outputted dependent variable points.
    pub y: Vec<SMatrix<T, R, C>>,

    /// Status of the solver.
    pub status: SolverStatus<T, R, C, E>,

    /// Number of function evaluations.
    pub evals: usize,

    /// Total number of steps taken by the solver.
    pub steps: usize,
    
    /// Number of rejected steps where the solution step-size had to be reduced.
    pub rejected_steps: usize,

    /// Number of accepted steps where the solution moved closer to tf.
    pub accepted_steps: usize,

    /// Time taken to solve the ODE in seconds.
    pub solve_time: T,
}

// Current Solution of the ODE Solver
impl<T, const R: usize, const C: usize, E> Solution<T, R, C, E> 
where 
    T: Real,
    E: EventData,
{
    /// Creates a new Solution object.
    /// 
    /// # Arguments
    /// * `solout` - The solution output object.
    /// 
    pub fn new() -> Self {
        Solution {
            t: Vec::with_capacity(100),
            y: Vec::with_capacity(100),
            status: SolverStatus::Uninitialized,
            evals: 0,
            steps: 0,
            rejected_steps: 0,
            accepted_steps: 0,
            solve_time: T::zero(),
        }
    }

    /// Puhes a new point to the solution, e.g. t and y vecs.
    /// 
    /// # Arguments
    /// * `t` - The time point.
    /// * `y` - The state vector.
    /// 
    pub fn push(&mut self, t: T, y: SMatrix<T, R, C>) {
        self.t.push(t);
        self.y.push(y);
    }

    /// Pops the last point from the solution, e.g. t and y vecs.
    /// 
    /// # Returns
    /// * `Option<(T, SMatrix<T, R, C>)>` - The last point in the solution.
    /// 
    pub fn pop(&mut self) -> Option<(T, SMatrix<T, R, C>)> {
        if self.t.is_empty() || self.y.is_empty() {
            return None;
        }
        let t = self.t.pop().unwrap();
        let y = self.y.pop().unwrap();
        Some((t, y))
    }

    /// Truncates the solution's (t, y) points to the given index.
    /// 
    /// # Arguments
    /// * `index` - The index to truncate to.
    /// 
    pub fn truncate(&mut self, index: usize) {
        self.t.truncate(index);
        self.y.truncate(index);
    }
}

// Post-processing methods for the solution
impl<T, const R: usize, const C: usize, E> Solution<T, R, C, E> 
where 
    T: Real,
    E: EventData,
{
    /// Simplifies the Solution into a tuple of vectors in form (t, y).
    /// By doing so, the Solution will be consumed and the status,
    /// evals, steps, rejected_steps, and accepted_steps will be discarded.
    /// 
    /// # Returns
    /// * `(Vec<T>, Vec<V)` - Tuple of time and state vectors.
    /// 
    pub fn into_tuple(self) -> (Vec<T>, Vec<SMatrix<T, R, C>>) {
        (self.t, self.y)
    }

    /// Returns the last accepted step of the solution in form (t, y).
    /// 
    /// # Returns
    /// * `Result<(T, V), Box<dyn std::error::Error>>` - Result of time and state vector.
    /// 
    pub fn last(&self) -> Result<(&T, &SMatrix<T, R, C>), Box<dyn std::error::Error>> {
        let t = self.t.last().ok_or("No t steps available")?;
        let y = self.y.last().ok_or("No y vectors available")?;
        Ok((t, y))
    }

    /// Returns an iterator over the solution.
    ///
    /// # Returns
    /// * `std::iter::Zip<std::slice::Iter<'_, T>, std::slice::Iter<'_, V>>` - An iterator
    ///   yielding (t, y) tuples.
    /// 
    pub fn iter(&self) -> std::iter::Zip<std::slice::Iter<'_, T>, std::slice::Iter<'_, SMatrix<T, R, C>>> {
        self.t.iter().zip(self.y.iter())
    }

    /// Creates a CSV file of the solution using standard library functionality.
    /// 
    /// Note the columns will be named t, y0, y1, ..., yN.
    /// 
    /// # Arguments
    /// * `filename` - Name of the file to save the solution.
    /// 
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - Result of writing the file.
    /// 
    #[cfg(not(feature = "polars"))]
    pub fn to_csv(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        
        // Create file and path if it does not exist
        let path = std::path::Path::new(filename);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file = std::fs::File::create(filename)?;

        // Length of state vector
        let n = self.y[0].len();
        
        // Write header
        let mut header = String::from("t");
        for i in 0..n {
            header.push_str(&format!(",y{}", i));
        }
        writeln!(file, "{}", header)?;
        
        // Write data
        for (t, y) in self.iter() {
            let mut row = format!("{:?}", t);
            for i in 0..n {
                row.push_str(&format!(",{:?}", y[i]));
            }
            writeln!(file, "{}", row)?;
        }
        
        Ok(())
    }

    /// Creates a csv file of the solution using Polars DataFrame.
    /// 
    /// Note the columns will be named t, y0, y1, ..., yN.
    /// 
    /// # Arguments
    /// * `filename` - Name of the file to save the solution.
    /// 
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - Result of writing the file.
    /// 
    #[cfg(feature = "polars")]
    pub fn to_csv(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create file and path if it does not exist
        let path = std::path::Path::new(filename);
        if !path.exists() {
            std::fs::create_dir_all(path.parent().unwrap())?;
        }
        let mut file = std::fs::File::create(filename)?;

        let t = self.t.iter().map(|x| x.to_f64()).collect::<Vec<f64>>();
        let mut columns = vec![Column::new("t".into(), t)];
        let n = self.y[0].len();
        for i in 0..n {
            let header = format!("y{}", i);
            columns.push(Column::new(header.into(), self.y.iter().map(|x| x[i].to_f64()).collect::<Vec<f64>>()));
        }
        let mut df = DataFrame::new(columns)?;

        // Write the dataframe to a csv file
        CsvWriter::new(&mut file).finish(&mut df)?;

        Ok(())
    }
    
    /// Creates a Polars DataFrame of the solution.
    /// 
    /// Note that the columns will be named t, y0, y1, ..., yN.
    /// 
    /// # Returns
    /// * `Result<DataFrame, PolarsError>` - Result of creating the DataFrame.
    /// 
    #[cfg(feature = "polars")]
    pub fn to_polars(&self) -> Result<DataFrame, PolarsError> {
        let t = self.t.iter().map(|x| x.to_f64()).collect::<Vec<f64>>();
        let mut columns = vec![Column::new("t".into(), t)];
        let n = self.y[0].len();
        for i in 0..n {
            let header = format!("y{}", i);
            columns.push(Column::new(header.into(), self.y.iter().map(|x| x[i].to_f64()).collect::<Vec<f64>>()));
        }

        DataFrame::new(columns)
    }
}

/// Interface for the solution to be used with the solout trait.
pub trait SolutionInterface<T, const R: usize, const C: usize, E> 
where 
    T: Real,
    E: EventData,
{
    /// Record the given point in the solution.
    fn record(&mut self, t: T, y: SMatrix<T, R, C>);

    /// Return the current step number.
    fn step(&self) -> usize;

    /// Return accepted steps.
    fn accepted_steps(&self) -> usize;

    /// Return rejected steps.
    fn rejected_steps(&self) -> usize;
}

impl <T, const R: usize, const C: usize, E> SolutionInterface<T, R, C, E> for Solution<T, R, C, E> 
where 
    T: Real,
    E: EventData,
{
    fn record(&mut self, t: T, y: SMatrix<T, R, C>) {
        self.push(t, y);
    }

    fn step(&self) -> usize {
        self.steps
    }

    fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }

    fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }
}