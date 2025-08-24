//! Tolerance enum for adaptive step size control

use crate::traits::Real;

use std::{
    convert::From,
    ops::{Index, IndexMut},
};

pub enum Tolerance<T: Real> {
    Scalar(T),
    Vector(Vec<T>),
}

impl<T: Real> Tolerance<T> {
    /// Average Tolerance
    pub fn average(&self) -> T {
        match self {
            Tolerance::Scalar(val) => *val,
            Tolerance::Vector(vec) => {
                let mut sum = T::zero();
                for i in vec {
                    sum += *i;
                }
                sum / T::from_usize(vec.len()).unwrap()
            }
        }
    }
}

impl<T: Real> Index<usize> for Tolerance<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        match self {
            Tolerance::Scalar(val) => val,
            Tolerance::Vector(vec) => &vec[i],
        }
    }
}

impl<T: Real> IndexMut<usize> for Tolerance<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        match self {
            Tolerance::Scalar(val) => val,
            Tolerance::Vector(vec) => &mut vec[i],
        }
    }
}

impl<T: Real> From<T> for Tolerance<T> {
    fn from(v: T) -> Self {
        Tolerance::Scalar(v)
    }
}

impl<T: Real> From<Vec<T>> for Tolerance<T> {
    fn from(v: Vec<T>) -> Self {
        Tolerance::Vector(v)
    }
}

impl<const N: usize, T: Real> From<[T; N]> for Tolerance<T> {
    fn from(v: [T; N]) -> Self {
        Tolerance::Vector(v.to_vec())
    }
}
