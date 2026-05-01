use crate::systems::{chaotic::*, linear::*, oscillators::*};
use criterion::{BenchmarkId, Criterion, criterion_group};
use differential_equations::{
    ivp::IVP,
    methods::{AdamsPredictorCorrector, ExplicitRungeKutta},
};
use nalgebra::vector;
use std::hint::black_box;

pub mod adaptive_step;
pub mod fixed_step;
