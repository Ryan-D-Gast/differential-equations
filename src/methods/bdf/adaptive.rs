use crate::{
    error::Error,
    interpolate::Interpolation,
    linalg::{Matrix, lin_solve, lu_decomp},
    methods::{Ordinary, h_init::InitialStepSize},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

use super::{BDF, BDF_ROWS, MAX_ORDER};

impl<T: Real, Y: State<T>> BDF<Ordinary, T, Y> {
    fn scalar(value: f64) -> T {
        T::from_f64(value).expect("BDF constants must be representable as the solver scalar type")
    }

    fn order_scalar(order: usize) -> T {
        T::from_usize(order)
            .expect("BDF order constants must be representable as the solver scalar type")
    }

    fn init_coefficients(&mut self) {
        let kappa = [
            T::zero(),
            Self::scalar(-0.1850),
            -T::one() / Self::scalar(9.0),
            Self::scalar(-0.0823),
            Self::scalar(-0.0415),
            T::zero(),
        ];

        self.gamma[0] = T::zero();
        for order in 1..=MAX_ORDER {
            self.gamma[order] = self.gamma[order - 1] + T::one() / Self::order_scalar(order);
            self.alpha[order] = (T::one() - kappa[order]) * self.gamma[order];
            self.error_const[order] =
                kappa[order] * self.gamma[order] + T::one() / Self::order_scalar(order + 1);
        }
    }

    fn weighted_rms_norm(&self, value: &Y, scale: &Y) -> T {
        let mut sum = T::zero();
        for i in 0..value.len() {
            let scaled = value.get(i) / scale.get(i);
            sum += scaled * scaled;
        }
        (sum / Self::order_scalar(value.len())).sqrt()
    }

    fn scale_from(&self, y: &Y) -> Y {
        let mut scale = Y::zeros();
        for i in 0..y.len() {
            scale.set(i, self.atol[i] + self.rtol[i] * y.get(i).abs());
        }
        scale
    }

    fn predict(&self, order: usize) -> Y {
        let mut y_predict = Y::zeros();
        for i in 0..=order {
            y_predict += self.d[i];
        }
        y_predict
    }

    fn compute_psi(&self, order: usize) -> Y {
        let mut psi = Y::zeros();
        for i in 1..=order {
            psi += self.d[i] * self.gamma[i];
        }
        psi / self.alpha[order]
    }

    fn compute_r(order: usize, factor: T) -> [[T; BDF_ROWS]; BDF_ROWS] {
        let mut r = [[T::zero(); BDF_ROWS]; BDF_ROWS];
        for j in 0..=order {
            r[0][j] = T::one();
        }

        for i in 1..=order {
            for j in 1..=order {
                r[i][j] = (Self::order_scalar(i - 1) - factor * Self::order_scalar(j))
                    / Self::order_scalar(i);
            }
        }

        for i in 1..=order {
            for j in 1..=order {
                r[i][j] *= r[i - 1][j];
            }
        }

        r
    }

    fn change_d(&mut self, order: usize, factor: T) {
        if factor == T::one() {
            return;
        }

        let r = Self::compute_r(order, factor);
        let u = Self::compute_r(order, T::one());
        let mut ru = [[T::zero(); BDF_ROWS]; BDF_ROWS];

        for i in 0..=order {
            for j in 0..=order {
                let mut sum = T::zero();
                for (k, u_row) in u.iter().enumerate().take(order + 1) {
                    sum += r[i][k] * u_row[j];
                }
                ru[i][j] = sum;
            }
        }

        let old = self.d;
        for i in 0..=order {
            let mut transformed = Y::zeros();
            for j in 0..=order {
                transformed += old[j] * ru[j][i];
            }
            self.d[i] = transformed;
        }
    }

    fn factor_matrix(&mut self, c: T) -> bool {
        let dim = self.y.len();
        for i in 0..dim {
            for j in 0..dim {
                self.newton_matrix[(i, j)] = if i == j {
                    T::one() - c * self.jacobian[(i, j)]
                } else {
                    -c * self.jacobian[(i, j)]
                };
            }
        }

        lu_decomp(&mut self.newton_matrix, &mut self.ip).is_ok()
    }

    fn solve_bdf_system<F>(
        &mut self,
        ode: &F,
        t_new: T,
        y_predict: Y,
        c: T,
        psi: Y,
        scale: Y,
        evals: &mut Evals,
    ) -> (bool, usize, Y, Y)
    where
        F: ODE<T, Y>,
    {
        let mut d = Y::zeros();
        let mut y = y_predict;
        let mut dy_norm_old: Option<T> = None;

        for iter in 0..self.max_newton_iter {
            ode.diff(t_new, &y, &mut self.dydt);
            evals.function += 1;

            let mut dy = self.dydt * c - psi - d;
            lin_solve(&self.newton_matrix, &mut dy, &self.ip);
            let dy_norm = self.weighted_rms_norm(&dy, &scale);

            let rate = dy_norm_old.map(|old| dy_norm / old);
            if let Some(rate) = rate {
                let remaining = self.max_newton_iter - iter;
                if rate >= T::one()
                    || rate.powf(Self::order_scalar(remaining)) / (T::one() - rate) * dy_norm
                        > self.newton_tol
                {
                    return (false, iter + 1, y, d);
                }
            }

            y += dy;
            d += dy;

            if dy_norm == T::zero()
                || rate.is_some_and(|rate| rate / (T::one() - rate) * dy_norm < self.newton_tol)
            {
                return (true, iter + 1, y, d);
            }

            dy_norm_old = Some(dy_norm);
        }

        (false, self.max_newton_iter, y, d)
    }

    fn step_too_small(&self) -> bool {
        let min_step = self
            .h_min
            .max(T::default_epsilon() * Self::scalar(10.0) * (self.t.abs() + T::one()));
        self.h.abs() < min_step
    }

    fn apply_step_factor(&mut self, order: usize, factor: T) {
        self.h = constrain_step_size(self.h * factor, self.h_min, self.h_max);
        self.h = (self.filter)(self.h);
        self.change_d(order, factor);
        self.n_equal_steps = 0;
        self.lu_valid = false;
    }
}

impl<T: Real, Y: State<T>> OrdinaryNumericalMethod<T, Y> for BDF<Ordinary, T, Y> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        if self.h0 == T::zero() {
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, 1, &self.rtol, &self.atol, self.h_min, self.h_max, &mut evals,
            );
        }

        self.h = validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf)?;
        self.h = (self.filter)(self.h);
        self.init_coefficients();
        if self.newton_tol == T::zero() {
            let rtol = self.rtol.average();
            self.newton_tol = (Self::scalar(10.0) * T::default_epsilon() / rtol)
                .max(Self::scalar(0.03).min(rtol.sqrt()));
        }

        self.t = t0;
        self.y = *y0;
        self.t_prev = t0;
        self.y_prev = *y0;
        self.h_prev = self.h;
        self.order = 1;
        self.n_equal_steps = 0;
        self.steps = 0;
        self.lu_valid = false;

        ode.diff(t0, y0, &mut self.dydt);
        evals.function += 1;

        let dim = y0.len();
        self.d = [Y::zeros(); BDF_ROWS];
        self.d[0] = *y0;
        self.d[1] = self.dydt * self.h;
        self.jacobian = Matrix::zeros(dim, dim);
        self.newton_matrix = Matrix::zeros(dim, dim);
        self.ip = vec![0; dim];

        ode.jacobian(t0, y0, &mut self.jacobian);
        evals.jacobian += 1;

        self.status = Status::Initialized;
        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        if self.step_too_small() {
            let e = Error::StepSize {
                t: self.t,
                y: self.y,
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        if self.steps >= self.max_steps {
            let e = Error::MaxSteps {
                t: self.t,
                y: self.y,
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        let mut order = self.order;
        let mut step_accepted = false;
        let mut accepted_y = self.y;
        let mut accepted_d = Y::zeros();
        let mut accepted_error_norm = T::zero();
        let mut accepted_safety = T::one();

        while !step_accepted {
            if self.step_too_small() {
                let e = Error::StepSize {
                    t: self.t,
                    y: self.y,
                };
                self.status = Status::Error(e.clone());
                return Err(e);
            }

            let t_new = self.t + self.h;
            let y_predict = self.predict(order);
            let scale_predict = self.scale_from(&y_predict);
            let psi = self.compute_psi(order);
            let c = self.h / self.alpha[order];

            let mut converged = false;
            let mut n_iter = self.max_newton_iter;
            let mut y_new = y_predict;
            let mut d = Y::zeros();
            let mut jacobian_current = false;

            while !converged {
                if !self.lu_valid {
                    if !self.factor_matrix(c) {
                        converged = false;
                        break;
                    }
                    self.lu_valid = true;
                }

                (converged, n_iter, y_new, d) =
                    self.solve_bdf_system(ode, t_new, y_predict, c, psi, scale_predict, &mut evals);

                if !converged {
                    if jacobian_current {
                        break;
                    }
                    ode.jacobian(t_new, &y_predict, &mut self.jacobian);
                    evals.jacobian += 1;
                    self.lu_valid = false;
                    jacobian_current = true;
                }
            }

            if !converged {
                self.apply_step_factor(order, Self::scalar(0.5));
                continue;
            }

            let safety = Self::scalar(0.9) * Self::order_scalar(2 * self.max_newton_iter + 1)
                / Self::order_scalar(2 * self.max_newton_iter + n_iter);
            let scale = self.scale_from(&y_new);
            let error = d * self.error_const[order];
            let error_norm = self.weighted_rms_norm(&error, &scale);

            if error_norm > T::one() {
                let factor = self
                    .min_scale
                    .max(safety * error_norm.powf(-T::one() / Self::order_scalar(order + 1)));
                self.apply_step_factor(order, factor);
            } else {
                step_accepted = true;
                accepted_y = y_new;
                accepted_d = d;
                accepted_error_norm = error_norm;
                accepted_safety = safety;
            }
        }

        self.t_prev = self.t;
        self.y_prev = self.y;
        self.h_prev = self.h;
        self.t += self.h;
        self.y = accepted_y;
        self.steps += 1;
        self.n_equal_steps += 1;

        self.d[order + 2] = accepted_d - self.d[order + 1];
        self.d[order + 1] = accepted_d;
        for i in (0..=order).rev() {
            self.d[i] += self.d[i + 1];
        }

        if self.n_equal_steps > order {
            let scale = self.scale_from(&self.y);
            let error_m_norm = if order > 1 {
                self.weighted_rms_norm(&(self.d[order] * self.error_const[order - 1]), &scale)
            } else {
                T::infinity()
            };
            let error_p_norm = if order < self.max_order {
                self.weighted_rms_norm(&(self.d[order + 2] * self.error_const[order + 1]), &scale)
            } else {
                T::infinity()
            };

            let current_factor =
                accepted_error_norm.powf(-T::one() / Self::order_scalar(order + 1));
            let lower_factor = error_m_norm.powf(-T::one() / Self::order_scalar(order));
            let higher_factor = error_p_norm.powf(-T::one() / Self::order_scalar(order + 2));

            let mut best_factor = current_factor;
            let mut delta_order: isize = 0;
            if lower_factor > best_factor {
                best_factor = lower_factor;
                delta_order = -1;
            }
            if higher_factor > best_factor {
                best_factor = higher_factor;
                delta_order = 1;
            }

            order = (order as isize + delta_order) as usize;
            self.order = order;

            let factor = self.max_scale.min(accepted_safety * best_factor);
            self.apply_step_factor(order, factor);
        }

        self.status = Status::Solving;
        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &Y {
        &self.y
    }

    fn t_prev(&self) -> T {
        self.t_prev
    }

    fn y_prev(&self) -> &Y {
        &self.y_prev
    }

    fn h(&self) -> T {
        self.h
    }

    fn set_h(&mut self, h: T) {
        let h = (self.filter)(h);
        if self.h != T::zero() {
            let factor = h / self.h;
            self.change_d(self.order, factor);
            self.n_equal_steps = 0;
            self.lu_valid = false;
        }
        self.h = h;
    }

    fn status(&self) -> &Status<T, Y> {
        &self.status
    }

    fn set_status(&mut self, status: Status<T, Y>) {
        self.status = status;
    }
}

impl<T: Real, Y: State<T>> Interpolation<T, Y> for BDF<Ordinary, T, Y> {
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        if self.h_prev == T::zero() {
            return Ok(self.y);
        }

        let mut y_interp = self.d[0];
        let mut product = T::one();
        for j in 1..=self.order {
            let shift = self.t - self.h_prev * Self::order_scalar(j - 1);
            let denom = self.h_prev * Self::order_scalar(j);
            product = product * (t_interp - shift) / denom;
            y_interp += self.d[j] * product;
        }

        Ok(y_interp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use nalgebra::{SVector, vector};

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
    fn bdf_tracks_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf: BDF<Ordinary, f64, SVector<f64, 1>> = BDF::adaptive().rtol(1e-7).atol(1e-9);

        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0])
            .method(bdf)
            .solve()
            .unwrap();

        let actual = results.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn bdf_can_run_at_order_one() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf: BDF<Ordinary, f64, SVector<f64, 1>> =
            BDF::adaptive().max_order(1).rtol(1e-7).atol(1e-9);

        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0])
            .method(bdf)
            .solve()
            .unwrap();

        let actual = results.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (actual - expected).abs() < 1e-4,
            "got {actual}, expected {expected}"
        );
    }
}
