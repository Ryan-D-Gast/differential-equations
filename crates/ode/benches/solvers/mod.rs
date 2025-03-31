use criterion::{black_box, criterion_group, Criterion, BenchmarkId};
use nalgebra::vector;
use ode::*;
use ode::solvers::*;
use crate::systems::{
    linear::*,
    oscillators::*,
    chaotic::*,
};

//pub mod solutions;
pub mod fixed_step;
pub mod adaptive_step;