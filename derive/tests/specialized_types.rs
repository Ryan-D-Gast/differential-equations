use differential_equations_derive::State;
use differential_equations::traits::{State as StateTrait};
use nalgebra::{Vector2, Vector3, Vector4, Vector5, Vector6, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6, RowVector2, RowVector3, SMatrix};
use num_complex::Complex;

#[test]
fn test_all_vector_types() {
    #[derive(State)]
    struct AllVectors<T> {
        v2: Vector2<T>,
        v3: Vector3<T>,
        v4: Vector4<T>,
        v5: Vector5<T>,
        v6: Vector6<T>,
    }

    let mut state = AllVectors {
        v2: Vector2::new(1.0, 2.0),
        v3: Vector3::new(3.0, 4.0, 5.0),
        v4: Vector4::new(6.0, 7.0, 8.0, 9.0),
        v5: Vector5::new(10.0, 11.0, 12.0, 13.0, 14.0),
        v6: Vector6::new(15.0, 16.0, 17.0, 18.0, 19.0, 20.0),
    };
    
    // Total elements: 2 + 3 + 4 + 5 + 6 = 20
    assert_eq!(state.len(), 20);
    
    // Test first and last elements
    assert_eq!(state.get(0), 1.0);   // v2[0]
    assert_eq!(state.get(1), 2.0);   // v2[1]
    assert_eq!(state.get(19), 20.0); // v6[5]
    
    // Test setting values
    state.set(0, 100.0);
    assert_eq!(state.get(0), 100.0);
    
    // Test zeros
    let zero_state = AllVectors::<f64>::zeros();
    for i in 0..20 {
        assert_eq!(zero_state.get(i), 0.0);
    }
}

#[test]
fn test_all_matrix_types() {
    #[derive(State)]
    struct AllMatrices<T> {
        m2: Matrix2<T>,
        m3: Matrix3<T>,
        m4: Matrix4<T>,
        m5: Matrix5<T>,
        m6: Matrix6<T>,
    }

    let state = AllMatrices {
        m2: Matrix2::from_row_slice(&[1.0, 2.0, 3.0, 4.0]),
        m3: Matrix3::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
        m4: Matrix4::from_diagonal(&Vector4::new(14.0, 15.0, 16.0, 17.0)),
        m5: Matrix5::from_diagonal(&Vector5::new(18.0, 19.0, 20.0, 21.0, 22.0)),
        m6: Matrix6::from_diagonal(&Vector6::new(23.0, 24.0, 25.0, 26.0, 27.0, 28.0)),
    };
    
    // Total elements: 4 + 9 + 16 + 25 + 36 = 90
    assert_eq!(state.len(), 90);
    
    // Test some specific elements
    assert_eq!(state.get(0), 1.0);   // m2[(0,0)]
    assert_eq!(state.get(1), 2.0);   // m2[(0,1)]
    assert_eq!(state.get(2), 3.0);   // m2[(1,0)]
    assert_eq!(state.get(3), 4.0);   // m2[(1,1)]
    
    // Test zeros initialization
    let zero_state = AllMatrices::<f64>::zeros();
    for i in 0..90 {
        assert_eq!(zero_state.get(i), 0.0);
    }
}

#[test]
fn test_row_vectors() {
    #[derive(State)]
    struct RowVectors<T> {
        rv2: RowVector2<T>,
        rv3: RowVector3<T>,
    }

    let state = RowVectors {
        rv2: RowVector2::new(1.0, 2.0),
        rv3: RowVector3::new(3.0, 4.0, 5.0),
    };
    
    // Total elements: 2 + 3 = 5
    assert_eq!(state.len(), 5);
    
    assert_eq!(state.get(0), 1.0); // rv2[0]
    assert_eq!(state.get(1), 2.0); // rv2[1]
    assert_eq!(state.get(2), 3.0); // rv3[0]
    assert_eq!(state.get(3), 4.0); // rv3[1]
    assert_eq!(state.get(4), 5.0); // rv3[2]
}

#[test]
fn test_explicit_smatrix_types() {
    #[derive(State)]
    struct ExplicitSMatrices<T> {
        sm_2x3: SMatrix<T, 2, 3>,
        sm_3x2: SMatrix<T, 3, 2>,
        sm_1x5: SMatrix<T, 1, 5>,
        sm_5x1: SMatrix<T, 5, 1>,
    }

    let state = ExplicitSMatrices {
        sm_2x3: SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        sm_3x2: SMatrix::<f64, 3, 2>::from_row_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        sm_1x5: SMatrix::<f64, 1, 5>::from_row_slice(&[13.0, 14.0, 15.0, 16.0, 17.0]),
        sm_5x1: SMatrix::<f64, 5, 1>::from_row_slice(&[18.0, 19.0, 20.0, 21.0, 22.0]),
    };
    
    // Total elements: 6 + 6 + 5 + 5 = 22
    assert_eq!(state.len(), 22);
    
    // Test some specific elements
    assert_eq!(state.get(0), 1.0);   // sm_2x3[(0,0)]
    assert_eq!(state.get(5), 6.0);   // sm_2x3[(1,2)]
    assert_eq!(state.get(21), 22.0); // sm_5x1[(4,0)]
}

#[test]
fn test_mixed_nalgebra_comprehensive() {
    #[derive(State)]
    struct MixedNalgebra<T> {
        vec: Vector3<T>,
        row: RowVector2<T>,
        mat: Matrix2<T>,
        smat: SMatrix<T, 2, 4>,
    }

    let state = MixedNalgebra {
        vec: Vector3::new(1.0, 2.0, 3.0),
        row: RowVector2::new(4.0, 5.0),
        mat: Matrix2::new(6.0, 7.0, 8.0, 9.0),
        smat: SMatrix::<f64, 2, 4>::from_row_slice(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
    };
    
    // Total elements: 3 + 2 + 4 + 8 = 17
    assert_eq!(state.len(), 17);
    
    // Test arithmetic operations work with mixed nalgebra types
    let other = MixedNalgebra {
        vec: Vector3::new(1.0, 1.0, 1.0),
        row: RowVector2::new(1.0, 1.0),
        mat: Matrix2::new(1.0, 1.0, 1.0, 1.0),
        smat: SMatrix::<f64, 2, 4>::from_element(1.0),
    };
    
    let sum = state + other;
    assert_eq!(sum.vec[0], 2.0);
    assert_eq!(sum.row[0], 5.0);
    assert_eq!(sum.mat[(0,0)], 7.0);
    assert_eq!(sum.smat[(0,0)], 11.0);
}

#[test]
fn test_complex_array_combinations() {
    #[derive(State)]
    struct ComplexArrays<T> {
        complex_array: [Complex<T>; 3],
        mixed_array: [T; 2],
        single_complex: Complex<T>,
    }

    let state = ComplexArrays {
        complex_array: [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ],
        mixed_array: [7.0, 8.0],
        single_complex: Complex::new(9.0, 10.0),
    };
    
    // Total elements: 3*2 + 2 + 2 = 10 (3 complex numbers = 6 reals, 2 reals, 1 complex = 2 reals)
    assert_eq!(state.len(), 10);
    
    // Test complex array element access
    assert_eq!(state.get(0), 1.0); // complex_array[0].re
    assert_eq!(state.get(1), 2.0); // complex_array[0].im
    assert_eq!(state.get(2), 3.0); // complex_array[1].re
    assert_eq!(state.get(3), 4.0); // complex_array[1].im
    assert_eq!(state.get(4), 5.0); // complex_array[2].re
    assert_eq!(state.get(5), 6.0); // complex_array[2].im
    assert_eq!(state.get(6), 7.0); // mixed_array[0]
    assert_eq!(state.get(7), 8.0); // mixed_array[1]
    assert_eq!(state.get(8), 9.0); // single_complex.re
    assert_eq!(state.get(9), 10.0); // single_complex.im
}
