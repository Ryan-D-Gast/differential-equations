//! Integration tests for the State derive macro
//! 
//! This test suite comprehensively tests all supported field types and their combinations:
//! 1. Single scalar fields (T)
//! 2. Array fields ([T; N])
//! 3. Nalgebra matrix/vector fields (Vector2-6, Matrix2-6, RowVector2-6, SMatrix<T,R,C>)
//! 4. Complex number fields (Complex<T>)
//! 
//! The tests verify:
//! - Proper element counting and indexing
//! - Get/set operations for all field types
//! - Arithmetic operations (Add, Sub, Mul, Div, AddAssign)
//! - Zero initialization
//! - Edge cases and error conditions
//! - Debug formatting
//! - Clone/Copy traits

mod basic_tests;
mod comprehensive_tests;
mod edge_cases;
mod specialized_types;

use differential_equations_derive::State;
use differential_equations::traits::{State as StateTrait};

/// Quick smoke test to ensure the derive macro works at all
#[test]
fn smoke_test() {
    #[derive(State)]
    struct SimpleState<T> {
        x: T,
        y: T,
    }

    let mut state = SimpleState { x: 1.0, y: 2.0 };
    
    // Basic functionality
    assert_eq!(state.len(), 2);
    assert_eq!(state.get(0), 1.0);
    assert_eq!(state.get(1), 2.0);
    
    state.set(0, 10.0);
    assert_eq!(state.get(0), 10.0);
    
    // Arithmetic
    let other = SimpleState { x: 5.0, y: 3.0 };
    let sum = state + other;
    assert_eq!(sum.x, 15.0);
    assert_eq!(sum.y, 5.0);
    
    // Zeros
    let zero = SimpleState::<f64>::zeros();
    assert_eq!(zero.x, 0.0);
    assert_eq!(zero.y, 0.0);
}

/// Test that demonstrates the real-world usage scenario
#[test]
fn real_world_usage_example() {
    use nalgebra::Vector3;
    use num_complex::Complex;
    
    /// Example state vector for a complex physical system
    #[derive(State)]
    struct PhysicsState<T> {
        // Position and velocity in 3D
        position: Vector3<T>,
        velocity: Vector3<T>,
        
        // Some scalar properties
        temperature: T,
        pressure: T,
        
        // Array of sensor readings
        sensor_data: [T; 5],
        
        // Complex impedance for electrical properties
        impedance: Complex<T>,
    }
    
    let mut physics_state = PhysicsState {
        position: Vector3::new(1.0, 2.0, 3.0),
        velocity: Vector3::new(0.1, 0.2, 0.3),
        temperature: 273.15,
        pressure: 101325.0,
        sensor_data: [1.1, 2.2, 3.3, 4.4, 5.5],
        impedance: Complex::new(50.0, 25.0),
    };
    
    // This state should have: 3 + 3 + 1 + 1 + 5 + 2 = 15 elements
    assert_eq!(physics_state.len(), 15);
    
    // We should be able to integrate this with an ODE solver
    // by accessing individual elements and performing arithmetic
    
    // Simulate some dynamics: update position based on velocity
    for i in 0..3 {
        let pos = physics_state.get(i);
        let vel = physics_state.get(i + 3);
        physics_state.set(i, pos + vel * 0.1); // dt = 0.1
    }
    
    // Verify the update worked
    assert!((physics_state.get(0) - 1.01f64).abs() < 1e-10);
    assert!((physics_state.get(1) - 2.02f64).abs() < 1e-10);
    assert!((physics_state.get(2) - 3.03f64).abs() < 1e-10);
}

/// Performance test to ensure the generated code is reasonably efficient
#[test]
fn performance_test() {
    use nalgebra::{Vector3, Matrix3};
    
    #[derive(State)]
    struct LargeState<T> {
        vectors: [Vector3<T>; 10],
        matrices: [Matrix3<T>; 5],
        scalars: [T; 100],
    }
    
    let mut state = LargeState::<f64>::zeros();
    
    // This should have: 10*3 + 5*9 + 100 = 30 + 45 + 100 = 175 elements
    assert_eq!(state.len(), 175);
    
    // Benchmark basic operations
    let start = std::time::Instant::now();
    
    // Perform many get/set operations
    for _ in 0..1000 {
        for i in 0..state.len() {
            let val = state.get(i);
            state.set(i, val + 1.0);
        }
    }
    
    let duration = start.elapsed();
    println!("Performance test completed in: {:?}", duration);
    
    // Verify the operations worked
    assert_eq!(state.get(0), 1000.0);
    assert_eq!(state.get(174), 1000.0);
}
