//! Adams Methods

use crate::interpolate::{InterpolationError, cubic_hermite_interpolate};
use crate::ode::solver::NumEvals;
use crate::ode::solvers::utils::{constrain_step_size, validate_step_size_parameters};
use crate::ode::{CallBackData, ODE, Solver, SolverError, SolverStatus};
use crate::traits::Real;
use nalgebra::SMatrix;

mod apcf4;
mod apcv4;

pub use apcf4::{
    APCF4, // Adams-Predictor-Corrector 4th Order Fixed Step Size Method
};

pub use apcv4::{
    APCV4, // Adams-Predictor-Corrector 4th Order Variable Step Size Method
};
