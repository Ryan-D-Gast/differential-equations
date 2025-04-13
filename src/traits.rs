//! Defines Generics for the library. Includes generics for the floating point numbers.

use nalgebra::RealField;

/// Real Number Trait
///
/// This trait specifies the acceptable types for real numbers.
/// Currently implemented for:
/// * `f32` - 32-bit floating point
/// * `f64` - 64-bit floating point
///
/// Provides additional functionality required for ODE solvers beyond
/// what's provided by nalgebra's RealField trait.
///
pub trait Real: Copy + RealField {
    fn infinity() -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    fn infinity() -> Self {
        std::f32::INFINITY
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Real for f64 {
    fn infinity() -> Self {
        std::f64::INFINITY
    }

    fn to_f64(self) -> f64 {
        self
    }
}

/// Callback data trait
///
/// This trait represents data that can be returned from functions
/// that are used to control the solver's execution flow. The
/// Clone and Debug traits are required for internal use but anything
/// that implements this trait can be used as callback data.
/// For example, this can be a string, a number, or any other type
/// that implements the Clone and Debug traits.
///
pub trait CallBackData: Clone + std::fmt::Debug {}

// Implement for any type that already satisfies the bounds
impl<T: Clone + std::fmt::Debug> CallBackData for T {}