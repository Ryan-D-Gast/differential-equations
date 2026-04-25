//! Linear algebra helper operations

use crate::traits::{Real, State};

pub fn dot<T, Y>(a: &Y, b: &Y) -> T
where
    T: Real,
    Y: State<T>,
{
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        let mut sum = T::zero();
        let len = a_slice.len();
        for i in 0..len {
            sum += a_slice[i] * b_slice[i];
        }
        sum
    } else {
        let mut sum = T::zero();
        for i in 0..a.len() {
            sum += a.get(i) * b.get(i);
        }
        sum
    }
}

pub fn norm<T, Y>(a: Y) -> T
where
    T: Real,
    Y: State<T>,
{
    if let Some(a_slice) = a.as_slice() {
        let mut sum = T::zero();
        let len = a_slice.len();
        for i in 0..len {
            sum += a_slice[i] * a_slice[i];
        }
        sum.sqrt()
    } else {
        let mut sum = T::zero();
        for i in 0..a.len() {
            sum += a.get(i) * a.get(i);
        }
        sum.sqrt()
    }
}

pub fn component_multiply<T, Y>(a: &Y, b: &Y) -> Y
where
    T: Real,
    Y: State<T>,
{
    let mut result = *a;
    if let (Some(res_slice), Some(a_slice), Some(b_slice)) =
        (result.as_mut_slice(), a.as_slice(), b.as_slice())
    {
        let len = a_slice.len();
        for i in 0..len {
            res_slice[i] = a_slice[i] * b_slice[i];
        }
    } else {
        for i in 0..a.len() {
            result.set(i, a.get(i) * b.get(i));
        }
    }
    result
}

pub fn component_square<T: Real, Y: State<T>>(v: &Y) -> Y {
    let mut result = Y::zeros();

    if let (Some(res_slice), Some(v_slice)) = (result.as_mut_slice(), v.as_slice()) {
        let len = v_slice.len();
        for i in 0..len {
            let val = v_slice[i];
            res_slice[i] = val * val;
        }
    } else {
        for i in 0..v.len() {
            let val = v.get(i);
            result.set(i, val * val);
        }
    }

    result
}
