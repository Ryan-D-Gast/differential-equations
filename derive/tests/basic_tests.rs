use differential_equations_derive::State;
use differential_equations::traits::{State as StateTrait};
use nalgebra::{Vector2, Vector3, Matrix2};
use num_complex::Complex;

// Test helper function to verify basic State trait functionality
fn test_state_basics<S: StateTrait<f64>>(state: &mut S, expected_len: usize) {
    // Test length
    assert_eq!(state.len(), expected_len);
    
    // Test get/set for all elements
    for i in 0..expected_len {
        let original = state.get(i);
        state.set(i, 42.0);
        assert_eq!(state.get(i), 42.0);
        state.set(i, original); // Restore
    }
}

#[test]
fn test_single_field_only() {
    #[derive(State, PartialEq)]
    struct SingleField<T> {
        x: T,
    }

    let mut state = SingleField { x: 1.0 };
    test_state_basics(&mut state, 1);
    
    // Test arithmetic operations
    let other = SingleField { x: 2.0 };
    let sum = state + other;
    assert_eq!(sum.x, 3.0);
    
    let difference = sum - SingleField { x: 1.0 };
    assert_eq!(difference.x, 2.0);
    
    let scaled = difference * 3.0;
    assert_eq!(scaled.x, 6.0);
    
    let divided = scaled / 2.0;
    assert_eq!(divided.x, 3.0);
    
    // Test zeros
    let zero = SingleField::<f64>::zeros();
    assert_eq!(zero.x, 0.0);
}

#[test]
fn test_multiple_single_fields() {
    #[derive(State)]
    struct MultipleFields<T> {
        x: T,
        y: T,
        z: T,
    }

    let mut state = MultipleFields { x: 1.0, y: 2.0, z: 3.0 };
    test_state_basics(&mut state, 3);
    
    // Test specific indexing
    assert_eq!(state.get(0), 1.0);
    assert_eq!(state.get(1), 2.0);
    assert_eq!(state.get(2), 3.0);
    
    // Test arithmetic
    let other = MultipleFields { x: 1.0, y: 1.0, z: 1.0 };
    let sum = state + other;
    assert_eq!(sum.x, 2.0);
    assert_eq!(sum.y, 3.0);
    assert_eq!(sum.z, 4.0);
}

#[test]
fn test_array_fields_only() {
    #[derive(State)]
    struct ArrayFields<T> {
        small_array: [T; 2],
        large_array: [T; 5],
    }

    let mut state = ArrayFields {
        small_array: [1.0, 2.0],
        large_array: [3.0, 4.0, 5.0, 6.0, 7.0],
    };
    test_state_basics(&mut state, 7); // 2 + 5 = 7 elements
    
    // Test array element access
    assert_eq!(state.get(0), 1.0); // small_array[0]
    assert_eq!(state.get(1), 2.0); // small_array[1]
    assert_eq!(state.get(2), 3.0); // large_array[0]
    assert_eq!(state.get(6), 7.0); // large_array[4]
    
    // Test arithmetic with arrays
    let other = ArrayFields {
        small_array: [0.5, 0.5],
        large_array: [1.0, 1.0, 1.0, 1.0, 1.0],
    };
    let sum = state + other;
    assert_eq!(sum.small_array[0], 1.5);
    assert_eq!(sum.large_array[2], 6.0);
}

#[test]
fn test_nalgebra_fields_only() {
    #[derive(State)]
    struct NalgebraFields<T> {
        vec2: Vector2<T>,
        vec3: Vector3<T>,
        mat2: Matrix2<T>,
    }

    let mut state = NalgebraFields {
        vec2: Vector2::new(1.0, 2.0),
        vec3: Vector3::new(3.0, 4.0, 5.0),
        mat2: Matrix2::new(6.0, 7.0, 8.0, 9.0),
    };
    test_state_basics(&mut state, 9); // 2 + 3 + 4 = 9 elements
    
    // Test matrix element access (row-major order)
    assert_eq!(state.get(0), 1.0); // vec2[0]
    assert_eq!(state.get(1), 2.0); // vec2[1]
    assert_eq!(state.get(2), 3.0); // vec3[0]
    assert_eq!(state.get(4), 5.0); // vec3[2]
    assert_eq!(state.get(5), 6.0); // mat2[(0,0)]
    assert_eq!(state.get(6), 7.0); // mat2[(0,1)]
    assert_eq!(state.get(7), 8.0); // mat2[(1,0)]
    assert_eq!(state.get(8), 9.0); // mat2[(1,1)]
}

#[test]
fn test_complex_fields_only() {
    #[derive(State)]
    struct ComplexFields<T> {
        z1: Complex<T>,
        z2: Complex<T>,
    }

    let mut state = ComplexFields {
        z1: Complex::new(1.0, 2.0),
        z2: Complex::new(3.0, 4.0),
    };
    test_state_basics(&mut state, 4); // 2 complex = 4 real elements
    
    // Test complex element access
    assert_eq!(state.get(0), 1.0); // z1.re
    assert_eq!(state.get(1), 2.0); // z1.im
    assert_eq!(state.get(2), 3.0); // z2.re
    assert_eq!(state.get(3), 4.0); // z2.im
    
    // Test complex arithmetic
    let other = ComplexFields {
        z1: Complex::new(0.5, 0.5),
        z2: Complex::new(1.0, 1.0),
    };
    let sum = state + other;
    assert_eq!(sum.z1.re, 1.5);
    assert_eq!(sum.z1.im, 2.5);
    assert_eq!(sum.z2.re, 4.0);
    assert_eq!(sum.z2.im, 5.0);
}

#[test]
fn test_mixed_field_types() {
    #[derive(State)]
    struct MixedState<T> {
        scalar: T,
        array: [T; 3],
        vector: Vector2<T>,
        complex: Complex<T>,
    }

    let mut state = MixedState {
        scalar: 1.0,
        array: [2.0, 3.0, 4.0],
        vector: Vector2::new(5.0, 6.0),
        complex: Complex::new(7.0, 8.0),
    };
    test_state_basics(&mut state, 8); // 1 + 3 + 2 + 2 = 8 elements
    
    // Test mixed element access
    assert_eq!(state.get(0), 1.0); // scalar
    assert_eq!(state.get(1), 2.0); // array[0]
    assert_eq!(state.get(3), 4.0); // array[2]
    assert_eq!(state.get(4), 5.0); // vector[0]
    assert_eq!(state.get(5), 6.0); // vector[1]
    assert_eq!(state.get(6), 7.0); // complex.re
    assert_eq!(state.get(7), 8.0); // complex.im
}
