use crate::{
    error::Error,
    linalg::{lu_decomp, lin_solve, Matrix},
    methods::{Fixed, Ordinary},
    ode::{OrdinaryNumericalMethod, ODE},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::validate_step_size_parameters,
};

use super::BackwardDifferentiationFormula;

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 1> {
    /// Backward Euler (BDF1) with fixed step size.
    pub fn bdf1(h: T) -> Self {
        Self {
            h0: h,
            gamma: T::one(),
            alpha: [T::one()],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 2> {
    /// BDF2 with fixed step size.
    pub fn bdf2(h: T) -> Self {
        let two = T::from_f64(2.0).unwrap();
        let three = T::from_f64(3.0).unwrap();
        let four = T::from_f64(4.0).unwrap();
        Self {
            h0: h,
            gamma: two / three,
            alpha: [four / three, -T::one() / three],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 3> {
    /// BDF3 with fixed step size.
    pub fn bdf3(h: T) -> Self {
        let two = T::from_f64(2.0).unwrap();
        let six = T::from_f64(6.0).unwrap();
        let nine = T::from_f64(9.0).unwrap();
        let eleven = T::from_f64(11.0).unwrap();
        let eighteen = T::from_f64(18.0).unwrap();
        Self {
            h0: h,
            gamma: six / eleven,
            alpha: [eighteen / eleven, -nine / eleven, two / eleven],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 4> {
    /// BDF4 with fixed step size.
    pub fn bdf4(h: T) -> Self {
        let three = T::from_f64(3.0).unwrap();
        let twelve = T::from_f64(12.0).unwrap();
        let sixteen = T::from_f64(16.0).unwrap();
        let twenty_five = T::from_f64(25.0).unwrap();
        let thirty_six = T::from_f64(36.0).unwrap();
        let forty_eight = T::from_f64(48.0).unwrap();
        Self {
            h0: h,
            gamma: twelve / twenty_five,
            alpha: [forty_eight / twenty_five, -thirty_six / twenty_five, sixteen / twenty_five, -three / twenty_five],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 5> {
    /// BDF5 with fixed step size.
    pub fn bdf5(h: T) -> Self {
        let twelve = T::from_f64(12.0).unwrap();
        let sixty = T::from_f64(60.0).unwrap();
        let seventy_five = T::from_f64(75.0).unwrap();
        let one_thirty_seven = T::from_f64(137.0).unwrap();
        let two_hundred = T::from_f64(200.0).unwrap();
        let three_hundred = T::from_f64(300.0).unwrap();
        Self {
            h0: h,
            gamma: sixty / one_thirty_seven,
            alpha: [
                three_hundred / one_thirty_seven,
                -three_hundred / one_thirty_seven,
                two_hundred / one_thirty_seven,
                -seventy_five / one_thirty_seven,
                twelve / one_thirty_seven,
            ],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>> BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, 6> {
    /// BDF6 with fixed step size.
    pub fn bdf6(h: T) -> Self {
        let ten = T::from_f64(10.0).unwrap();
        let sixty = T::from_f64(60.0).unwrap();
        let seventy_two = T::from_f64(72.0).unwrap();
        let one_forty_seven = T::from_f64(147.0).unwrap();
        let two_twenty_five = T::from_f64(225.0).unwrap();
        let three_sixty = T::from_f64(360.0).unwrap();
        let four_hundred = T::from_f64(400.0).unwrap();
        let four_fifty = T::from_f64(450.0).unwrap();
        Self {
            h0: h,
            gamma: sixty / one_forty_seven,
            alpha: [
                three_sixty / one_forty_seven,
                -four_fifty / one_forty_seven,
                four_hundred / one_forty_seven,
                -two_twenty_five / one_forty_seven,
                seventy_two / one_forty_seven,
                -ten / one_forty_seven,
            ],
            ..Default::default()
        }
    }
}

impl<T: Real, Y: State<T>, const O: usize> OrdinaryNumericalMethod<T, Y>
    for BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, O>
{
    fn init<F>(&mut self, _ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let evals = Evals::new();

        match validate_step_size_parameters::<T, Y>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h) => self.h = h,
            Err(e) => return Err(e),
        }

        self.t = t0;
        self.y = *y0;

        for i in 0..O {
            self.t_prev[i] = t0 - self.h * T::from_usize(i + 1).unwrap();
            self.y_prev[i] = *y0; // Ideally use proper initialization with RK4 or similar, but padding with y0 for now
        }

        let dim = y0.len();
        self.jacobian = Matrix::zeros(dim, dim);
        self.newton_matrix = Matrix::zeros(dim, dim);
        self.rhs_newton = vec![T::zero(); dim];
        self.delta_y_vec = vec![T::zero(); dim];

        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();
        let dim = self.y.len();

        if self.steps >= self.max_steps {
            let e = Error::MaxSteps { t: self.t, y: self.y };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        let t_next = self.t + self.h;
        let mut y_next = self.y; // Predictor: y_{n+1} = y_n

        let mut ip = vec![0; dim];

        // Newton iteration
        for iter in 0..self.max_newton_iter {
            ode.diff(t_next, &y_next, &mut self.dydt);
            evals.function += 1;

            if iter == 0 || self.jacobian_age > 10 {
                ode.jacobian(t_next, &y_next, &mut self.jacobian);
                evals.jacobian += 1;
                self.jacobian_evaluations += 1;
                self.jacobian_age = 0;

                // M = I - h * gamma * J
                for i in 0..dim {
                    for j in 0..dim {
                        self.newton_matrix[(i, j)] = if i == j {
                            T::one() - self.h * self.gamma * self.jacobian[(i, j)]
                        } else {
                            -self.h * self.gamma * self.jacobian[(i, j)]
                        };
                    }
                }

                if let Err(e) = lu_decomp(&mut self.newton_matrix, &mut ip) {
                    let err = Error::LinearAlgebra { msg: e.to_string() };
                    self.status = Status::Error(err.clone());
                    return Err(err);
                }
                self.lu_decompositions += 1;
            }

            let mut history_sum = Y::zeros();
            // alpha[0] corresponds to y_n
            // alpha[1] corresponds to y_{n-1}
            // ...
            history_sum += self.y * self.alpha[0];
            for i in 1..O {
                history_sum += self.y_prev[i - 1] * self.alpha[i];
            }

            let residual = y_next - history_sum - self.dydt * (self.h * self.gamma);
            let mut res_norm = T::zero();
            for i in 0..dim {
                self.rhs_newton[i] = -residual.get(i);
                res_norm += residual.get(i).powi(2);
            }
            res_norm = res_norm.sqrt();

            if res_norm < self.newton_tol * (self.h.abs() * self.gamma) {
                break;
            }

            let mut dy_y = Y::zeros();
            for i in 0..dim {
                dy_y.set(i, self.rhs_newton[i]);
            }

            lin_solve(&self.newton_matrix, &mut dy_y, &ip);

            let mut dy_norm = T::zero();
            for i in 0..dim {
                let dy = dy_y.get(i);
                dy_norm += dy.powi(2);
                y_next.set(i, y_next.get(i) + dy);
            }
            dy_norm = dy_norm.sqrt();

            if dy_norm < self.newton_tol {
                break;
            }

            if iter == self.max_newton_iter - 1 {
                let e = Error::StepSize { t: self.t, y: self.y };
                self.status = Status::Error(e.clone());
                return Err(e);
            }
        }

        self.jacobian_age += 1;

        // Shift history: y_{n-O+1} is dropped.
        for i in (1..O).rev() {
            self.t_prev[i] = self.t_prev[i - 1];
            self.y_prev[i] = self.y_prev[i - 1];
        }
        self.t_prev[0] = self.t;
        self.y_prev[0] = self.y;

        self.t = t_next;
        self.y = y_next;
        self.steps += 1;

        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &Y {
        &self.y
    }

    fn t_prev(&self) -> T {
        self.t_prev[0]
    }

    fn y_prev(&self) -> &Y {
        &self.y_prev[0]
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        self.h = h;
    }

    fn status(&self) -> &Status<T, Y> {
        &self.status
    }

    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

// interpolation module trait for Dense output
use crate::interpolate::Interpolation;
impl<T: Real, Y: State<T>, const O: usize> Interpolation<T, Y>
    for BackwardDifferentiationFormula<Ordinary, Fixed, T, Y, O>
{
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        // Implement Hermite interpolation for dense output using current and previous state
        let h = self.t - self.t_prev[0];
        if h == T::zero() {
            return Ok(self.y); // Return current if no step taken
        }

        if t_interp < self.t_prev[0] || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev[0],
                t_curr: self.t,
            });
        }

        let theta = (t_interp - self.t_prev[0]) / h;
        // Simple linear interpolation as fallback, or use better for higher order
        let mut y_interp = Y::zeros();
        for i in 0..self.y.len() {
            let y_prev_val = self.y_prev[0].get(i);
            let y_curr_val = self.y.get(i);
            y_interp.set(i, y_prev_val + theta * (y_curr_val - y_prev_val));
        }

        Ok(y_interp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use nalgebra::{vector, SVector};

    struct ExponentialDecay {
        k: f64,
    }

    impl ODE<f64, SVector<f64, 1>> for ExponentialDecay {
        fn diff(&self, _t: f64, y: &SVector<f64, 1>, dydt: &mut SVector<f64, 1>) {
            dydt[0] = -self.k * y[0];
        }

        fn jacobian(&self, _t: f64, _y: &SVector<f64, 1>, j: &mut Matrix<f64>) {
            j[(0, 0)] = -self.k;
        }
    }

    #[test]
    fn test_bdf1_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf1 = BackwardDifferentiationFormula::bdf1(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf1).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3855).abs() < 0.01, "BDF1 failed. Actual: {}", actual_val);
    }

    #[test]
    fn test_bdf2_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf2 = BackwardDifferentiationFormula::bdf2(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf2).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3678_f64).abs() < 0.05, "BDF2 failed. Actual: {}", actual_val);
    }

    #[test]
    fn test_bdf3_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf3 = BackwardDifferentiationFormula::bdf3(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf3).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3678_f64).abs() < 0.05, "BDF3 failed. Actual: {}", actual_val);
    }

    #[test]
    fn test_bdf4_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf4 = BackwardDifferentiationFormula::bdf4(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf4).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3678_f64).abs() < 0.05, "BDF4 failed. Actual: {}", actual_val);
    }

    #[test]
    fn test_bdf5_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf5 = BackwardDifferentiationFormula::bdf5(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf5).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3678_f64).abs() < 0.05, "BDF5 failed. Actual: {}", actual_val);
    }

    #[test]
    fn test_bdf6_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf6 = BackwardDifferentiationFormula::bdf6(0.1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0]).method(bdf6).solve().unwrap();
        let actual_val = results.y.last().unwrap()[0];
        assert!((actual_val - 0.3678_f64).abs() < 0.05, "BDF6 failed. Actual: {}", actual_val);
    }
}
