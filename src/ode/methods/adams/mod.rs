//! Adams Methods

pub use crate::{
    Error, Status,
    traits::{Real, CallBackData},
    interpolate::{InterpolationError, cubic_hermite_interpolate},
    ode::{
        ODE, NumericalMethod, NumEvals,
        methods::utils::{constrain_step_size, validate_step_size_parameters},
    },
};
use nalgebra::SMatrix;

mod apcf4;
mod apcv4;

pub use apcf4::{
    APCF4, // Adams-Predictor-Corrector 4th Order Fixed Step Size Method
};

pub use apcv4::{
    APCV4, // Adams-Predictor-Corrector 4th Order Variable Step Size Method
};
