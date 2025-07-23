use differential_equations_derive::State;
use differential_equations::traits::{State as StateTrait};
use nalgebra::{Vector2, Vector3, Matrix2, SMatrix};
use num_complex::Complex;

#[test]
fn test_comprehensive_mixed_state() {
    /// The most comprehensive test - uses all 4 supported field types
    #[derive(State)]
    struct ComprehensiveState<T> {
        // Single fields
        scalar1: T,
        scalar2: T,
        
        // Array fields of different sizes
        small_array: [T; 2],
        medium_array: [T; 4],
        
        // Various nalgebra types
        vec2: Vector2<T>,
        vec3: Vector3<T>,
        mat2: Matrix2<T>,
        smatrix: SMatrix<T, 2, 3>, // Explicit SMatrix
        
        // Complex fields
        complex1: Complex<T>,
        complex2: Complex<T>,
    }

    let mut state = ComprehensiveState {
        scalar1: 1.0,
        scalar2: 2.0,
        small_array: [3.0, 4.0],
        medium_array: [5.0, 6.0, 7.0, 8.0],
        vec2: Vector2::new(9.0, 10.0),
        vec3: Vector3::new(11.0, 12.0, 13.0),
        mat2: Matrix2::new(14.0, 15.0, 16.0, 17.0),
        smatrix: SMatrix::<f64, 2, 3>::new(18.0, 19.0, 20.0, 21.0, 22.0, 23.0),
        complex1: Complex::new(24.0, 25.0),
        complex2: Complex::new(26.0, 27.0),
    };
    
    // Total elements: 2 + 2 + 4 + 2 + 3 + 4 + 6 + 2 + 2 = 27
    assert_eq!(state.len(), 27);
    
    // Test get/set for all elements
    for i in 0..27 {
        let original = state.get(i);
        state.set(i, 42.0);
        assert_eq!(state.get(i), 42.0);
        state.set(i, original); // Restore
    }
    
    // Test that zeros work for comprehensive state
    let zero_state = ComprehensiveState::<f64>::zeros();
    assert_eq!(zero_state.scalar1, 0.0);
    assert_eq!(zero_state.small_array[0], 0.0);
    assert_eq!(zero_state.vec2[0], 0.0);
    assert_eq!(zero_state.complex1.re, 0.0);
    assert_eq!(zero_state.complex1.im, 0.0);
}

#[test]
fn test_arithmetic_operations_comprehensive() {
    #[derive(State)]
    struct ArithmeticTest<T> {
        scalar: T,
        array: [T; 2],
        vector: Vector2<T>,
        complex: Complex<T>,
    }

    let state1 = ArithmeticTest {
        scalar: 2.0,
        array: [4.0, 6.0],
        vector: Vector2::new(8.0, 10.0),
        complex: Complex::new(12.0, 14.0),
    };

    let state2 = ArithmeticTest {
        scalar: 1.0,
        array: [2.0, 3.0],
        vector: Vector2::new(4.0, 5.0),
        complex: Complex::new(6.0, 7.0),
    };

    // Test Add
    let sum = state1 + state2;
    assert_eq!(sum.scalar, 3.0);
    assert_eq!(sum.array[0], 6.0);
    assert_eq!(sum.array[1], 9.0);
    assert_eq!(sum.vector[0], 12.0);
    assert_eq!(sum.vector[1], 15.0);
    assert_eq!(sum.complex.re, 18.0);
    assert_eq!(sum.complex.im, 21.0);

    // Test Sub
    let diff = state1 - state2;
    assert_eq!(diff.scalar, 1.0);
    assert_eq!(diff.array[0], 2.0);
    assert_eq!(diff.array[1], 3.0);
    assert_eq!(diff.vector[0], 4.0);
    assert_eq!(diff.vector[1], 5.0);
    assert_eq!(diff.complex.re, 6.0);
    assert_eq!(diff.complex.im, 7.0);

    // Test Mul (scalar multiplication)
    let scaled = state1 * 2.0;
    assert_eq!(scaled.scalar, 4.0);
    assert_eq!(scaled.array[0], 8.0);
    assert_eq!(scaled.array[1], 12.0);
    assert_eq!(scaled.vector[0], 16.0);
    assert_eq!(scaled.vector[1], 20.0);
    assert_eq!(scaled.complex.re, 24.0);
    assert_eq!(scaled.complex.im, 28.0);

    // Test Div (scalar division)
    let divided = scaled / 4.0;
    assert_eq!(divided.scalar, 1.0);
    assert_eq!(divided.array[0], 2.0);
    assert_eq!(divided.array[1], 3.0);
    assert_eq!(divided.vector[0], 4.0);
    assert_eq!(divided.vector[1], 5.0);
    assert_eq!(divided.complex.re, 6.0);
    assert_eq!(divided.complex.im, 7.0);

    // Test AddAssign
    let mut mutable_state = state1;
    mutable_state += state2;
    assert_eq!(mutable_state.scalar, 3.0);
    assert_eq!(mutable_state.array[0], 6.0);
    assert_eq!(mutable_state.complex.re, 18.0);
}

#[test]
fn test_various_nalgebra_types() {
    #[derive(State)]
    struct VariousNalgebra<T> {
        vec2: Vector2<T>,
        vec3: Vector3<T>,
        mat2: Matrix2<T>,
        smat23: SMatrix<T, 2, 3>,
        smat31: SMatrix<T, 3, 1>,
    }

    let mut state = VariousNalgebra {
        vec2: Vector2::new(1.0, 2.0),
        vec3: Vector3::new(3.0, 4.0, 5.0),
        mat2: Matrix2::new(6.0, 7.0, 8.0, 9.0),
        smat23: SMatrix::<f64, 2, 3>::new(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        smat31: SMatrix::<f64, 3, 1>::new(16.0, 17.0, 18.0),
    };
    
    // Total: 2 + 3 + 4 + 6 + 3 = 18 elements
    assert_eq!(state.len(), 18);
    
    // Test get/set for all elements
    for i in 0..18 {
        let original = state.get(i);
        state.set(i, 42.0);
        assert_eq!(state.get(i), 42.0);
        state.set(i, original); // Restore
    }
}
