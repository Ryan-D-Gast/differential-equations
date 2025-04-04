//! Hyperplane crossing detection for finding when a trajectory intersects a hyperplane.
//!
//! This module allows detection and recording of points where a solution trajectory
//! crosses a hyperplane in the state space.

use super::*;

/// Function type for extracting position components from state vector
pub type ExtractorFn<V, P> = fn(&V) -> P;

/// A solout that detects when a trajectory crosses a hyperplane.
/// 
/// # Overview
///
/// `HyperplaneSolout` monitors a trajectory and detects when it crosses a specified
/// hyperplane. This is useful for:
///
/// - Poincaré section analysis
/// - Detecting orbital events (e.g., equatorial crossings)
/// - Section-to-section mapping for dynamical systems
///
/// # Type Parameters
///
/// * `T`: Floating-point type
/// * `P`: Vector type for the position space (e.g., Vector3<f64>)
/// * `V`: Full state vector type (e.g., Vector6<f64>)
///
/// # Example
///
/// ```
/// use differential_equations::ode::*;
/// use nalgebra::{Vector3, Vector6, vector};
///
/// // CR3BP system (simplified representation)
/// struct CR3BP { mu: f64 }
///
/// impl ODE<f64, 6, 1> for CR3BP {
///     fn diff(&self, _t: f64, y: &Vector6<f64>, dydt: &mut Vector6<f64>) {
///     // Mass ratio
///     let mu = self.mu;
///
///     // Extracting states
///     let (rx, ry, rz, vx, vy, vz) = (y[0], y[1], y[2], y[3], y[4], y[5]);
///
///     // Distance to primary body
///     let r13 = ((rx + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();
///     // Distance to secondary body
///     let r23 = ((rx - 1.0 + mu).powi(2) + ry.powi(2) + rz.powi(2)).sqrt();
///
///     // Computing three-body dynamics
///     dydt[0] = vx;
///     dydt[1] = vy;
///     dydt[2] = vz;
///     dydt[3] = rx + 2.0 * vy - (1.0 - mu) * (rx + mu) / r13.powi(3) - mu * (rx - 1.0 + mu) / r23.powi(3);
///     dydt[4] = ry - 2.0 * vx - (1.0 - mu) * ry / r13.powi(3) - mu * ry / r23.powi(3);
///     dydt[5] = -(1.0 - mu) * rz / r13.powi(3) - mu * rz / r23.powi(3);
///     }
/// }
///
/// // Create the system
/// let system = CR3BP { mu: 0.012155 }; // Earth-Moon system
/// let t0 = 0.0;
/// let tf = 10.0;
/// let y0 = vector![ // 9:2 L2 Southern NRHO orbit
///     1.021881345465263, 0.0, -0.182000000000000, // Position
///     0.0, -0.102950816739606, 0.0 // Velocity
/// ];
/// let mut solver = DOP853::new().rtol(1e-12).atol(1e-12);
///
/// // Function to extract position from state vector
/// fn extract_position(state: &Vector6<f64>) -> Vector3<f64> {
///     vector![state[3], state[4], state[5]]
/// }
///
/// // Detect z=0 plane crossings (equatorial plane)
/// let plane_point = vector![1.0, 0.0, 0.0]; // Point on the plane
/// let plane_normal = vector![0.0, 1.0, 1.0]; // Normal vector (z-axis)
///
/// // Solve and get only the plane crossing points
/// let ivp = IVP::new(system, t0, tf, y0);
/// let solution = ivp.hyperplane_crossing(plane_point, plane_normal, extract_position, CrossingDirection::Both).solve(&mut solver).unwrap();
///
/// // solution now contains only the points where the trajectory crosses the z=0 plane
/// ```
pub struct HyperplaneCrossingSolout<T, const R1: usize, const C1: usize, const R2: usize, const C2: usize>
where
    T: Real,
{
    /// Point on the hyperplane
    point: SMatrix<T, R1, C1>,
    /// Normal vector to the hyperplane (should be normalized)
    normal: SMatrix<T, R1, C1>,
    /// Function to extract position components from state vector
    extractor: ExtractorFn<SMatrix<T, R2, C2>, SMatrix<T, R1, C1>>,
    /// Last observed signed distance (for detecting sign changes)
    last_distance: Option<T>,
    /// Direction of crossing to detect
    direction: CrossingDirection,
    /// Phantom data for state vector type
    _phantom: std::marker::PhantomData<SMatrix<T, R2, C2>>,
}

impl<T, const R1: usize, const C1: usize, const R2: usize, const C2: usize> HyperplaneCrossingSolout<T, R1, C1, R2, C2>
where
    T: Real,
{
    /// Creates a new HyperplaneSolout to detect when the trajectory crosses the specified hyperplane.
    ///
    /// By default, crossings in both directions are detected.
    ///
    /// # Arguments
    /// * `point` - A point on the hyperplane
    /// * `normal` - The normal vector to the hyperplane (will be normalized internally)
    /// * `extractor` - Function to extract position components from state vector
    ///
    /// # Returns
    /// * A new `HyperplaneCrossingSolout` instance
    /// 
    pub fn new(point: SMatrix<T, R1, C1>, mut normal: SMatrix<T, R1, C1>, extractor: ExtractorFn<SMatrix<T, R2, C2>, SMatrix<T, R1, C1>>) -> Self {
        // Normalize the normal vector
        let norm = normal.norm();
        if norm > T::default_epsilon() {
            normal = normal * T::one() / norm;
        }
        
        HyperplaneCrossingSolout {
            point,
            normal,
            extractor,
            last_distance: None,
            direction: CrossingDirection::Both,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Set the direction of crossings to detect.
    ///
    /// # Arguments
    /// * `direction` - The crossing direction to detect (Both, Positive, or Negative)
    ///
    /// # Returns
    /// * `Self` - The modified HyperplaneSolout (builder pattern)
    /// 
    pub fn with_direction(mut self, direction: CrossingDirection) -> Self {
        self.direction = direction;
        self
    }
    
    /// Set to detect only positive crossings (from negative to positive side).
    ///
    /// A positive crossing occurs when the trajectory transitions from the
    /// negative side to the positive side of the hyperplane, as defined by the
    /// normal vector.
    ///
    /// # Returns
    /// * `Self` - The modified HyperplaneSolout (builder pattern)
    /// 
    pub fn positive_only(mut self) -> Self {
        self.direction = CrossingDirection::Positive;
        self
    }
    
    /// Set to detect only negative crossings (from positive to negative side).
    ///
    /// A negative crossing occurs when the trajectory transitions from the
    /// positive side to the negative side of the hyperplane, as defined by the
    /// normal vector.
    ///
    /// # Returns
    /// * `Self` - The modified HyperplaneSolout (builder pattern)
    /// 
    pub fn negative_only(mut self) -> Self {
        self.direction = CrossingDirection::Negative;
        self
    }
    
    /// Calculate signed distance from a point to the hyperplane.
    ///
    /// # Arguments
    /// * `pos` - The position to calculate distance for
    ///
    /// # Returns
    /// * Signed distance (positive if on same side as normal vector)
    /// 
    fn signed_distance(&self, pos: &SMatrix<T, R1, C1>) -> T {
        // Calculate displacement vector from plane point to position
        let displacement = pos - self.point;
        
        // Dot product with normal gives signed distance
        displacement.dot(&self.normal)
    }
}

impl<T, const R1: usize, const C1: usize, const R2: usize, const C2: usize, E: EventData> Solout<T, R2, C2, E> for HyperplaneCrossingSolout<T, R1, C1, R2, C2>
where 
    T: Real,
    E: EventData
{
    fn solout<S, F>(&mut self, solver: &mut S, _ode: &F, t_out: &mut Vec<T>, y_out: &mut Vec<SMatrix<T, R2, C2>>)
    where 
        F: ODE<T, R2, C2, E>,
        S: Solver<T, R2, C2, E>
    {
        let t_curr = solver.t();
        let y_curr = solver.y();
        
        // Extract position from current state and calculate distance
        let pos_curr = (self.extractor)(y_curr);
        let distance = self.signed_distance(&pos_curr);
        
        // If we have a previous distance, check for crossing
        if let Some(last_distance) = self.last_distance {
            let zero = T::zero();
            let is_crossing = last_distance.signum() != distance.signum() || 
                             (last_distance == zero && distance != zero) ||
                             (last_distance != zero && distance == zero);
            
            if is_crossing {
                // Check crossing direction if specified
                let record_crossing = match self.direction {
                    CrossingDirection::Positive => last_distance < zero && distance >= zero,
                    CrossingDirection::Negative => last_distance > zero && distance <= zero,
                    CrossingDirection::Both => true, // any crossing
                };
                
                if record_crossing {
                    let t_prev = solver.t_prev();
                    
                    // Extract position derivatives
                    let vel_prev = (self.extractor)(solver.dydt_prev());
                    let vel_curr = (self.extractor)(solver.dydt());
                    
                    // Find intersection time by finding when distance function equals zero
                    // For each component affected by the normal vector:
                    // Solve: point + normal * t = pos_prev + (pos_curr - pos_prev) * t
                    
                    // Calculate cubic hermite crossing for the signed distance function
                    // Get rates of change of the signed distance function
                    let dist_rate_prev = vel_prev.dot(&self.normal);
                    let dist_rate_curr = vel_curr.dot(&self.normal);
                    
                    // Find the time when the distance function equals zero
                    if let Some(t_cross) = find_cubic_hermite_crossing(
                        t_prev,
                        t_curr,
                        last_distance,
                        distance,
                        dist_rate_prev,
                        dist_rate_curr,
                        T::zero()
                    ) {
                        // Use solver's interpolation for the full state vector
                        let y_cross = solver.interpolate(t_cross).unwrap();
                        
                        // Record the crossing time and value
                        t_out.push(t_cross);
                        y_out.push(y_cross);
                    } else {
                        // Fallback to linear interpolation if cubic method fails
                        let frac = -last_distance / (distance - last_distance);
                        let t_cross = t_prev + frac * (t_curr - t_prev);
                        let y_cross = solver.interpolate(t_cross).unwrap();
                        
                        t_out.push(t_cross);
                        y_out.push(y_cross);
                    }
                }
            }
        }
        
        // Update last distance for next comparison
        self.last_distance = Some(distance);
    }

    fn include_t0_tf(&self) -> bool {
        false // Do not include t0 and tf in the output unless they are crossings
    }
}