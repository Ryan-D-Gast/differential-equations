//! Adams Methods

use nalgebra::SMatrix;
use crate::ode::{Solver, SolverStatus, System, EventData};
use crate::traits::Real;
use crate::ode::solvers::utils::{validate_step_size_parameters, constrain_step_size};

mod apcf4;
mod apcv4;

pub use apcf4::{
    APCF4,   // Adams-Predictor-Corrector 4th Order Fixed Step Size Method
};

pub use apcv4::{
    APCV4,   // Adams-Predictor-Corrector 4th Order Variable Step Size Method
};