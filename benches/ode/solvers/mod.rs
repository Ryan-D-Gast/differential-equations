use crate::systems::{chaotic::*, linear::*, oscillators::*};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group};
use differential_equations::ode::methods::*;
use differential_equations::ode::*;
use nalgebra::vector;

//pub mod solutions;
pub mod adaptive_step;
pub mod fixed_step;
