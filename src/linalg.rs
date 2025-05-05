//! Linear algebra helper operations

use crate::traits::{Real, State};

pub fn dot<T, V>(a: &V, b: &V) -> T
where
    T: Real,
    V: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * b.get(i);
    }
    sum
}

pub fn norm<T, V>(a: V) -> T
where
    T: Real,
    V: State<T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a.get(i) * a.get(i);
    }
    sum.sqrt()
}

pub fn component_multiply<T, V>(a: &V, b: &V) -> V
where
    T: Real,
    V: State<T>,
{
    let mut result = a.clone();
    for i in 0..a.len() {
        result.set(i, a.get(i) * b.get(i));
    }
    result
}

pub fn component_square<T: Real, V: State<T>>(v: &V) -> V {
    let mut result = V::zeros();
    
    for i in 0..v.len() {
        let val = v.get(i);
        result.set(i, val * val);
    }
    
    result
}