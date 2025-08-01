use crate::systems::{chaotic::*, linear::*, oscillators::*};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group};
use differential_equations::{
    methods::{AdamsPredictorCorrector, ExplicitRungeKutta},
    ode::*,
};
use nalgebra::vector;

pub mod adaptive_step;
pub mod fixed_step;
