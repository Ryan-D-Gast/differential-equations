//! Linear algebra helper operations

use crate::traits::{Real, State};

pub fn dot<T, Y>(a: &Y, b: &Y) -> T
where
    T: Real,
    Y: State<T>,
{
    a.dot(b)
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
    a.component_mul(b)
}

pub fn component_square<T: Real, Y: State<T>>(v: &Y) -> Y {
    v.component_mul(v)
}
