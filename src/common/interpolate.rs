//! Interpolation Methods for the IVP struct when solving the system.

use crate::traits::Real;
use nalgebra::SMatrix;

/// Cubic Hermite Interpolation
/// 
/// # Arguments
/// * `t0` - Initial Time.
/// * `t1` - Final Time.
/// * `y0` - Initial State Vector.
/// * `y1` - Final State Vector.
/// * `t`  - Time to interpolate at.
/// 
/// # Returns
/// * Interpolated State Vector.
/// 
pub fn cubic_hermite_interpolate<T: Real, const R: usize, const C: usize>(t0: T, t1: T, y0: &SMatrix<T, R, C>, y1: &SMatrix<T, R, C>, k0: &SMatrix<T, R, C>, k1: &SMatrix<T, R, C>, t: T) -> SMatrix<T, R, C>
{
    let two = T::from_f64(2.0).unwrap();
    let three = T::from_f64(3.0).unwrap();
    let h = t1 - t0;
    let s = (t - t0) / h;
    let h00 = two * s.powi(3) - three * s.powi(2) + T::one();
    let h10 = s.powi(3) - two * s.powi(2) + s;
    let h01 = -two * s.powi(3) + three * s.powi(2);
    let h11 = s.powi(3) - s.powi(2);
    y0 * h00 + k0 * h10 * h + y1 * h01 + k1 * h11 * h
}

/// Find the time at which a cubic Hermite polynomial crosses a threshold value
/// 
/// # Arguments
/// * `t0` - Initial time
/// * `t1` - Final time
/// * `y0` - Initial value
/// * `y1` - Final value
/// * `k0` - Initial derivative
/// * `k1` - Final derivative
/// * `threshold` - Value to find crossing of
/// 
/// # Returns
/// * Some(t) - Time at which polynomial crosses threshold, if found
/// * None - If no crossing exists or could not be reliably found
/// 
pub fn find_cubic_hermite_crossing<T: Real>(
    t0: T, 
    t1: T, 
    y0: T, 
    y1: T, 
    k0: T, 
    k1: T, 
    threshold: T
) -> Option<T> {
    // Check if endpoints straddle the threshold
    let f0 = y0 - threshold;
    let f1 = y1 - threshold;
    
    // If no crossing or exactly at an endpoint, handle those cases
    if f0 * f1 > T::zero() {
    // Same sign, no guaranteed crossing
    // We could still have an even number of crossings, so we'll check
    } else if f0 == T::zero() {
        return Some(t0);  // Exactly at start point
    } else if f1 == T::zero() {
        return Some(t1);  // Exactly at end point
    }
    
    // Normalize to [0,1] interval for numerical stability
    let h = t1 - t0;
    
    // Initial guess using linear interpolation
    let mut s = if f0 * f1 <= T::zero() {
        // Simple linear interpolation for initial guess if signs differ
        f0 / (f0 - f1)
    } else {
        // Start in middle if same sign (could have 0 or 2 crossings)
        T::from_f64(0.5).unwrap()
    };
    
    // Convert to normalized form
    let a = T::from_f64(2.0).unwrap() * (y0 - y1) + h * (k0 + k1);
    let b = T::from_f64(3.0).unwrap() * (y1 - y0) - h * (T::from_f64(2.0).unwrap() * k0 + k1);
    let c = h * k0;
    let d = y0 - threshold;
    
    // This is now a cubic equation: a*s³ + b*s² + c*s + d = 0
    // Use Newton's method with safeguards
    let max_iterations = 10;
    let tolerance = T::from_f64(1e-12).unwrap();
    
    for _ in 0..max_iterations {
        // Evaluate polynomial and its derivative
        let f_s = d + s * (c + s * (b + s * a));
        let df_ds = c + s * (T::from_f64(2.0).unwrap() * b + T::from_f64(3.0).unwrap() * a * s);
        
        // Check for convergence
        if f_s.abs() < tolerance {
            // Convert back to time
            return Some(t0 + s * h);
        }
        
        // Prevent division by zero
        if df_ds.abs() < T::from_f64(1e-10).unwrap() {
            // Try bisection or different approach if derivative is near zero
            break;
        }
        
        // Newton update
        let s_new = s - f_s / df_ds;
        
        // Check if new s is within [0,1]
        if s_new < T::zero() || s_new > T::one() {
            // If Newton's method steps outside valid range, use bisection
            if f0 * f1 <= T::zero() {
                // Only guaranteed to have a root if signs differ
                let mut a = T::zero();
                let mut b = T::one();
                let mut fa = f0;
                
                // Simple bisection method
                for _ in 0..10 {
                    let m = (a + b) / T::from_f64(2.0).unwrap();
                    let fm = d + m * (c + m * (b + m * a));
                    
                    if fm.abs() < tolerance {
                        return Some(t0 + m * h);
                    }
                    
                    if fa * fm <= T::zero() {
                        b = m;
                    } else {
                        a = m;
                        fa = fm;
                    }
                }
                
                // Return the best approximation
                return Some(t0 + ((a + b) / T::from_f64(2.0).unwrap()) * h);
            } else {
                // No guaranteed crossing
                return None;
            }
        }
        
        // Check for convergence
        if (s_new - s).abs() < tolerance {
            return Some(t0 + s_new * h);
        }
        
        s = s_new;
    }
    
    // If we exited the loop without returning, try a fallback approach
    if f0 * f1 <= T::zero() {
        // Use bisection as a last resort if we know there's a crossing
        let mut a = T::zero();
        let mut b = T::one();
        let mut fa = f0;
        
        for _ in 0..20 {
            let m = (a + b) / T::from_f64(2.0).unwrap();
            let s = m;
            let fm = d + s * (c + s * (b + s * a));
            
            if fm.abs() < tolerance {
                return Some(t0 + m * h);
            }
            
            if fa * fm <= T::zero() {
                b = m;
            } else {
                a = m;
                fa = fm;
            }
        }
        
        // Return our best approximation
        return Some(t0 + ((a + b) / T::from_f64(2.0).unwrap()) * h);
    }
    
    None  // No reliable crossing found
}