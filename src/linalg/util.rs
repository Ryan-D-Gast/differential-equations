//! Linear algebra helper operations

use crate::traits::{Real, State};

pub fn dot<T, Y>(a: &Y, b: &Y) -> T
where
    T: Real,
    Y: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * b.get(i);
    }
    sum
}

pub fn norm<T, Y>(a: Y) -> T
where
    T: Real,
    Y: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * a.get(i);
    }
    sum.sqrt()
}

pub fn component_multiply<T, Y>(a: &Y, b: &Y) -> Y
where
    T: Real,
    Y: State<T>,
{
    let mut result = *a;
    for i in 0..a.len() {
        result.set(i, a.get(i) * b.get(i));
    }
    result
}

pub fn component_square<T: Real, Y: State<T>>(v: &Y) -> Y {
    let mut result = Y::zeros();

    for i in 0..v.len() {
        let val = v.get(i);
        result.set(i, val * val);
    }

    result
}
