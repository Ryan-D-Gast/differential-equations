use differential_equations::traits::{Real, State as StateTrait};
use differential_equations_derive::State;
use nalgebra::{Matrix2, Vector2};
use num_complex::Complex;

trait TestStateAccess<T: Real>: StateTrait<T> {
    fn component(&self, index: usize) -> T {
        assert!(index < self.len(), "Index out of bounds");
        let mut values = vec![T::zero(); self.len()];
        self.write_to_slice(&mut values);
        values[index]
    }

    fn set_component(&mut self, index: usize, value: T) {
        assert!(index < self.len(), "Index out of bounds");
        let mut values = vec![T::zero(); self.len()];
        self.write_to_slice(&mut values);
        values[index] = value;
        self.read_from_slice(&values);
    }
}

impl<T, Y> TestStateAccess<T> for Y
where
    T: Real,
    Y: StateTrait<T>,
{
}

#[test]
fn test_edge_cases() {
    // Test single element array
    #[derive(State)]
    struct SingleElementArray<T> {
        single: [T; 1],
    }

    let mut state = SingleElementArray { single: [42.0] };
    assert_eq!(state.len(), 1);
    assert_eq!(state.component(0), 42.0);

    state.set_component(0, 100.0);
    assert_eq!(state.component(0), 100.0);

    // Test large array
    #[derive(State)]
    struct LargeArray<T> {
        large: [T; 10],
    }

    let mut large_state = LargeArray { large: [1.0; 10] };
    assert_eq!(large_state.len(), 10);

    for i in 0..10 {
        assert_eq!(large_state.component(i), 1.0);
        large_state.set_component(i, i as f64);
        assert_eq!(large_state.component(i), i as f64);
    }

    // Test unnamed fields (tuple struct)
    #[derive(State)]
    struct TupleStruct<T>(T, [T; 2], Vector2<T>);

    let tuple_state = TupleStruct(1.0, [2.0, 3.0], Vector2::new(4.0, 5.0));
    assert_eq!(tuple_state.len(), 5); // 1 + 2 + 2 = 5
    assert_eq!(tuple_state.component(0), 1.0);
    assert_eq!(tuple_state.component(1), 2.0);
    assert_eq!(tuple_state.component(4), 5.0);
}

#[test]
fn test_clone_and_copy() {
    #[derive(State, PartialEq)]
    struct CloneTest<T> {
        x: T,
        arr: [T; 2],
    }

    let original = CloneTest {
        x: 1.0,
        arr: [2.0, 3.0],
    };
    let cloned = original;
    let copied = original;

    assert_eq!(original, cloned);
    assert_eq!(original, copied);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_get_index_out_of_bounds() {
    #[derive(State)]
    struct SmallState<T> {
        x: T,
    }

    let state = SmallState { x: 1.0 };
    state.component(1); // Should panic
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_set_index_out_of_bounds() {
    #[derive(State)]
    struct SmallState<T> {
        x: T,
    }

    let mut state = SmallState { x: 1.0 };
    state.set_component(1, 42.0); // Should panic
}

#[test]
fn test_debug_formatting() {
    #[derive(State)]
    struct DebugTest<T> {
        scalar: T,
        array: [T; 2],
        complex: Complex<T>,
    }

    let state = DebugTest {
        scalar: 1.0,
        array: [2.0, 3.0],
        complex: Complex::new(4.0, 5.0),
    };

    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("DebugTest"));
    assert!(debug_str.contains("scalar"));
    assert!(debug_str.contains("array"));
    assert!(debug_str.contains("complex"));
}

#[test]
fn test_zeros_initialization() {
    #[derive(State)]
    struct ZerosTest<T> {
        scalar: T,
        array: [T; 3],
        vector: Vector2<T>,
        matrix: Matrix2<T>,
        complex: Complex<T>,
    }

    let zero_state = ZerosTest::<f64>::zeros();

    // Test that all elements are zero
    assert_eq!(zero_state.scalar, 0.0);
    assert_eq!(zero_state.array, [0.0, 0.0, 0.0]);
    assert_eq!(zero_state.vector, Vector2::new(0.0, 0.0));
    assert_eq!(zero_state.matrix, Matrix2::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(zero_state.complex, Complex::new(0.0, 0.0));

    // Test that length is correct
    assert_eq!(zero_state.len(), 12); // 1 + 3 + 2 + 4 + 2 = 12

    // Test that all indexed elements are zero
    for i in 0..zero_state.len() {
        assert_eq!(zero_state.component(i), 0.0);
    }
}

#[test]
fn test_different_numeric_types() {
    // Test with f32
    #[derive(State)]
    struct F32State<T> {
        x: T,
        y: T,
    }

    let state_f32 = F32State {
        x: 1.0f32,
        y: 2.0f32,
    };
    assert_eq!(state_f32.len(), 2);
    assert_eq!(state_f32.component(0), 1.0f32);
    assert_eq!(state_f32.component(1), 2.0f32);

    // Test zeros with f32
    let zero_f32 = F32State::<f32>::zeros();
    assert_eq!(zero_f32.x, 0.0f32);
    assert_eq!(zero_f32.y, 0.0f32);
}

#[test]
fn test_empty_arrays() {
    // Note: Arrays with size 0 are not common, but let's test size 1 as minimal case
    #[derive(State)]
    struct MinimalArray<T> {
        tiny: [T; 1],
        scalar: T,
    }

    let mut state = MinimalArray {
        tiny: [42.0],
        scalar: 1.0,
    };

    assert_eq!(state.len(), 2);
    assert_eq!(state.component(0), 42.0); // tiny[0]
    assert_eq!(state.component(1), 1.0); // scalar

    state.set_component(0, 100.0);
    assert_eq!(state.component(0), 100.0);
}

#[test]
fn test_complex_arithmetic_edge_cases() {
    #[derive(State)]
    struct ComplexArithmetic<T> {
        z: Complex<T>,
    }

    let state1 = ComplexArithmetic {
        z: Complex::new(3.0, 4.0),
    };
    let state2 = ComplexArithmetic {
        z: Complex::new(1.0, 2.0),
    };

    // Test complex addition
    let sum = state1 + state2;
    assert_eq!(sum.z.re, 4.0);
    assert_eq!(sum.z.im, 6.0);

    // Test complex subtraction
    let diff = state1 - state2;
    assert_eq!(diff.z.re, 2.0);
    assert_eq!(diff.z.im, 2.0);

    // Test complex scalar multiplication
    let scaled = state1 * 2.0;
    assert_eq!(scaled.z.re, 6.0);
    assert_eq!(scaled.z.im, 8.0);

    // Test complex scalar division
    let divided = scaled / 2.0;
    assert_eq!(divided.z.re, 3.0);
    assert_eq!(divided.z.im, 4.0);
}

#[test]
fn test_matrix_indexing_order() {
    #[derive(State)]
    struct MatrixIndexing<T> {
        mat: Matrix2<T>,
    }

    let state = MatrixIndexing {
        mat: Matrix2::new(1.0, 2.0, 3.0, 4.0),
    };

    // Matrix2::new(a, b, c, d) creates:
    // | a  b |
    // | c  d |
    // Our indexing should be row-major: (0,0), (0,1), (1,0), (1,1)
    assert_eq!(state.component(0), 1.0); // (0,0)
    assert_eq!(state.component(1), 2.0); // (0,1)
    assert_eq!(state.component(2), 3.0); // (1,0)
    assert_eq!(state.component(3), 4.0); // (1,1)
}
