//! Linear algebra helper operations

use crate::traits::{Real, State};

pub fn dot<T, Y>(a: &Y, b: &Y) -> T
where
    T: Real,
    Y: State<T>,
{
    assert_eq!(a.len(), b.len(), "State length mismatch");
    let mut a_values = vec![T::zero(); a.len()];
    let mut b_values = vec![T::zero(); b.len()];
    a.copy_to_flat_slice(&mut a_values);
    b.copy_to_flat_slice(&mut b_values);

    let mut sum = T::zero();
    for (ai, bi) in a_values.iter().zip(b_values.iter()) {
        sum += *ai * *bi;
    }
    sum
}

pub fn norm<T, Y>(a: Y) -> T
where
    T: Real,
    Y: State<T>,
{
    a.norm_squared().sqrt()
}

pub fn component_multiply<T, Y>(a: &Y, b: &Y) -> Y
where
    T: Real,
    Y: State<T>,
{
    assert_eq!(a.len(), b.len(), "State length mismatch");
    let mut result_values = vec![T::zero(); a.len()];
    let mut a_values = vec![T::zero(); a.len()];
    let mut b_values = vec![T::zero(); b.len()];
    a.copy_to_flat_slice(&mut a_values);
    b.copy_to_flat_slice(&mut b_values);

    for ((dst, ai), bi) in result_values
        .iter_mut()
        .zip(a_values.iter())
        .zip(b_values.iter())
    {
        *dst = *ai * *bi;
    }
    let mut result = a.zeros_like();
    result.copy_from_flat_slice(&result_values);
    result
}

pub fn component_square<T: Real, Y: State<T>>(v: &Y) -> Y {
    let mut values = vec![T::zero(); v.len()];
    v.copy_to_flat_slice(&mut values);

    for value in values.iter_mut() {
        *value *= *value;
    }

    let mut result = v.zeros_like();
    result.copy_from_flat_slice(&values);
    result
}
