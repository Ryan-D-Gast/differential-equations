//! Crossing detection solout for detecting when state components cross threshold values.
//!
//! This module provides functionality for detecting and recording when a specific state 
//! component crosses a defined threshold value during integration.

use super::*;

/// A solout that detects when a component crosses a specified threshold value.
/// 
/// # Overview
///
/// `CrossingSolout` monitors a specific component of the state vector and detects when
/// it crosses a defined threshold value. This is useful for identifying important events
/// in the system's behavior, such as:
///
/// - Zero-crossings (by setting threshold to 0)
/// - Detecting when a variable exceeds or falls below a critical value
/// - Generating data for poincare sections or other analyses
///
/// The solout records the times and states when crossings occur, making them available
/// in the solver output.
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
/// use differential_equations::ode::solout::CrossingSolout;
/// use nalgebra::{Vector2, vector};
///
/// // Simple harmonic oscillator - position will cross zero periodically
/// struct HarmonicOscillator;
///
/// impl ODE<f64, 2, 1> for HarmonicOscillator {
///     fn diff(&self, _t: f64, y: &Vector2<f64>, dydt: &mut Vector2<f64>) {
///         // y[0] = position, y[1] = velocity
///         dydt[0] = y[1];
///         dydt[1] = -y[0];
///     }
/// }
///
/// // Create the system and solver
/// let system = HarmonicOscillator;
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![1.0, 0.0]; // Start with positive position, zero velocity
/// let mut solver = DOP853::new().rtol(1e-8).atol(1e-8);
///
/// // Detect zero-crossings of the position component (index 0)
/// let mut crossing_detector = CrossingSolout::new(0, 0.0);
///
/// // Solve and get only the crossing points
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.solout(&mut crossing_detector).solve(&mut solver).unwrap();
///
/// // solution now contains only the points where position crosses zero
/// println!("Zero crossings occurred at times: {:?}", solution.t);
/// ```
///
/// # Directional Crossing Detection
///
/// You can filter the crossings by direction:
///
/// ```
/// use differential_equations::ode::solout::{CrossingSolout, CrossingDirection};
///
/// // Only detect positive crossings (from below to above threshold)
/// let positive_crossings = CrossingSolout::new(0, 5.0).with_direction(CrossingDirection::Positive);
///
/// // Only detect negative crossings (from above to below threshold)
/// let negative_crossings = CrossingSolout::new(0, 5.0).with_direction(CrossingDirection::Negative);
/// ```
pub struct CrossingSolout<T: Real> {
    /// Index of the component to monitor
    component_idx: usize,
    /// Threshold value to detect crossings against
    threshold: T,
    /// Last observed value minus threshold (for detecting sign changes)
    last_offset_value: Option<T>,
    /// Direction of crossing to detect
    direction: CrossingDirection,
}

impl<T: Real> CrossingSolout<T> {
    /// Creates a new CrossingSolout to detect when the specified component crosses the threshold.
    ///
    /// By default, crossings in both directions are detected.
    ///
    /// # Arguments
    /// * `component_idx` - Index of the component in the state vector to monitor
    /// * `threshold` - The threshold value to detect crossings against
    ///
    /// # Example
    ///
    /// ```
    /// use differential_equations::ode::solout::CrossingSolout;
    ///
    /// // Detect when the first component (index 0) crosses the value 5.0
    /// let detector = CrossingSolout::new(0, 5.0);
    /// ```
    pub fn new(component_idx: usize, threshold: T) -> Self {
        CrossingSolout {
            component_idx,
            threshold,
            last_offset_value: None,
            direction: CrossingDirection::Both,
        }
    }
    
    /// Set the direction of crossings to detect.
    ///
    /// # Arguments
    /// * `direction` - The crossing direction to detect (Both, Positive, or Negative)
    ///
    /// # Returns
    /// * `Self` - The modified CrossingSolout (builder pattern)
    ///
    /// # Example
    ///
    /// ```
    /// use differential_equations::ode::solout::{CrossingSolout, CrossingDirection};
    ///
    /// // Detect when the position (index 0) crosses zero in any direction
    /// let any_crossing = CrossingSolout::new(0, 0.0).with_direction(CrossingDirection::Both);
    ///
    /// // Detect when the position (index 0) goes from negative to positive
    /// let zero_up_detector = CrossingSolout::new(0, 0.0).with_direction(CrossingDirection::Positive);
    ///
    /// // Detect when the velocity (index 1) changes from positive to negative
    /// let velocity_sign_change = CrossingSolout::new(1, 0.0).with_direction(CrossingDirection::Negative);
    /// ```
    pub fn with_direction(mut self, direction: CrossingDirection) -> Self {
        self.direction = direction;
        self
    }
    
    /// Set to detect only positive crossings (from below to above threshold).
    ///
    /// A positive crossing occurs when the monitored component transitions from
    /// a value less than the threshold to a value greater than or equal to the threshold.
    ///
    /// # Returns
    /// * `Self` - The modified CrossingSolout (builder pattern)
    ///
    /// # Example
    ///
    /// ```
    /// use differential_equations::ode::solout::CrossingSolout;
    ///
    /// // Detect when the position (index 0) goes from negative to positive
    /// let zero_up_detector = CrossingSolout::new(0, 0.0).positive_only();
    /// ```
    pub fn positive_only(mut self) -> Self {
        self.direction = CrossingDirection::Positive;
        self
    }
    
    /// Set to detect only negative crossings (from above to below threshold).
    ///
    /// A negative crossing occurs when the monitored component transitions from
    /// a value greater than the threshold to a value less than or equal to the threshold.
    ///
    /// # Returns
    /// * `Self` - The modified CrossingSolout (builder pattern)
    ///
    /// # Example
    ///
    /// ```
    /// use differential_equations::ode::solout::CrossingSolout;
    ///
    /// // Detect when the velocity (index 1) changes from positive to negative
    /// let velocity_sign_change = CrossingSolout::new(1, 0.0).negative_only();
    /// ```
    pub fn negative_only(mut self) -> Self {
        self.direction = CrossingDirection::Negative;
        self
    }
}

impl<T, const R: usize, const C: usize, E> Solout<T, R, C, E> for CrossingSolout<T>
where 
    T: Real,
    E: EventData
{
    fn solout<SV, SI>(&mut self, solver: &mut SV, solution: &mut SI)
    where 
        SV: Solver<T, R, C, E>,
        SI: SolutionInterface<T, R, C, E> 
    {
        let t_curr = solver.t();
        let y_curr = solver.y();
        
        // Calculate the offset from threshold (to detect zero-crossing)
        let current_value = y_curr[self.component_idx];
        let offset_value = current_value - self.threshold;
        
        // If we have a previous value, check for crossing
        if let Some(last_offset) = self.last_offset_value {
            let zero = T::zero();
            let is_crossing = last_offset.signum() != offset_value.signum();
            
            if is_crossing {
                // Check crossing direction if specified
                let record_crossing = match self.direction {
                    CrossingDirection::Positive => last_offset < zero && offset_value >= zero,
                    CrossingDirection::Negative => last_offset > zero && offset_value <= zero,
                    CrossingDirection::Both => true, // any crossing
                };
                
                if record_crossing {
                    let t_prev = solver.t_prev();
                    
                    // Find crossing time using Newton's method
                    if let Some(t_cross) = self.find_crossing_newton(
                        solver, 
                        t_prev, 
                        t_curr, 
                        last_offset, 
                        offset_value
                    ) {
                        // Use solver's interpolation for the full state vector at crossing time
                        let y_cross = solver.interpolate(t_cross).unwrap();
                        
                        // Record the crossing time and value
                        solution.record(t_cross, y_cross);
                    } else {
                        // Fallback to linear interpolation if Newton's method fails
                        let frac = -last_offset / (offset_value - last_offset);
                        let t_cross = t_prev + frac * (t_curr - t_prev);
                        let y_cross = solver.interpolate(t_cross).unwrap();
                        
                        // Record the estimated crossing time and value
                        solution.record(t_cross, y_cross);
                    }
                }
            }
        }
        
        // Update last value for next comparison
        self.last_offset_value = Some(offset_value);
    }

    fn include_t0_tf(&self) -> bool {
        false // Do not include t0 and tf in the output
    }
}

// Add the Newton's method implementation
impl<T: Real> CrossingSolout<T> {
    /// Find the crossing time using Newton's method with solver interpolation
    fn find_crossing_newton<S, const R: usize, const C: usize, E>(
        &self, 
        solver: &mut S, 
        t_lower: T, 
        t_upper: T, 
        offset_lower: T, 
        offset_upper: T
    ) -> Option<T>
    where
        S: Solver<T, R, C, E>,
        E: EventData,
    {
        // Start with linear interpolation as initial guess
        let mut t = t_lower - offset_lower * (t_upper - t_lower) / (offset_upper - offset_lower);
        
        // Newton's method parameters
        let max_iterations = 10;
        let tolerance = T::default_epsilon() * T::from_f64(100.0).unwrap(); // Higher tolerance for numerical stability
        let mut offset;
        
        // Newton's method iterations
        for _ in 0..max_iterations {
            // Get interpolated state at current time guess
            let y_t = solver.interpolate(t).unwrap();
            
            // Calculate offset from threshold at this time point
            offset = y_t[self.component_idx] - self.threshold;
            
            // Check if we're close enough to the crossing
            if offset.abs() < tolerance {
                return Some(t);
            }
            
            // Calculate numerical derivative of the offset function
            let delta_t = (t_upper - t_lower) * T::from_f64(1e-6).unwrap();
            let t_plus = t + delta_t;
            let y_plus = solver.interpolate(t_plus).unwrap();
            let offset_plus = y_plus[self.component_idx] - self.threshold;
            
            let derivative = (offset_plus - offset) / delta_t;
            
            // Avoid division by zero or very small derivatives
            if derivative.abs() < T::default_epsilon() * T::from_f64(10.0).unwrap() {
                break;
            }
            
            // Newton step
            let t_new = t - offset / derivative;
            
            // Ensure we stay within bounds
            if t_new < t_lower || t_new > t_upper {
                // Bisection fallback
                t = (t_lower + t_upper) / T::from_f64(2.0).unwrap();
            } else {
                // Check if we're making progress
                let change = (t_new - t).abs();
                if change < tolerance * T::from_f64(0.1).unwrap() {
                    // We're barely moving, consider it converged
                    t = t_new;
                    break;
                }
                t = t_new;
            }
        }
        
        // Final check: Get interpolated value and see if we're close enough
        let y_t = solver.interpolate(t).unwrap();
        offset = y_t[self.component_idx] - self.threshold;
        
        if offset.abs() < tolerance * T::from_f64(10.0).unwrap() {
            Some(t)
        } else {
            None // Failed to converge
        }
    }
}