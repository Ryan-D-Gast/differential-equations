use crate::{
    error::Error,
    interpolate::Interpolation,
    linalg::{lu_decomp, Matrix},
    methods::{h_init::InitialStepSize, Ordinary},
    ode::{OrdinaryNumericalMethod, ODE},
    stats::Evals,
    status::Status,
    traits::{Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

use super::{Bdf, MAX_ORDER};

impl<T: Real, Y: State<T>> Bdf<Ordinary, T, Y> {
    fn compute_bdf_coefficients(&self, t_next: T) -> (T, [T; MAX_ORDER], usize) {
        let q = self.effective_order();
        if q == 0 {
            return (T::one(), [T::zero(); MAX_ORDER], 0);
        }

        // psi[j] = t_next - tau[j] where tau[0] = self.t, tau[j] = self.t_hist[j-1]
        let mut psi = [T::zero(); MAX_ORDER];
        psi[0] = t_next - self.t;
        for j in 1..q {
            psi[j] = t_next - self.t_hist[j - 1];
        }

        let h = psi[0];
        if h == T::zero() {
            return (T::one(), [T::zero(); MAX_ORDER], 0);
        }

        // l_0'(tau_0) = sum_{j=0}^{q-1} 1/psi[j]
        let l0_prime: T = (0..q).fold(T::zero(), |acc, j| acc + T::one() / psi[j]);

        // gamma: h * gamma = 1 / l_0'
        let gamma = T::one() / (h * l0_prime);

        // alpha[j] = -l_{j+1}'(tau_0) / l_0'(tau_0)
        // l_{j+1}'(tau_0) = -1/psi[j] * prod_{k=0,k!=j}^{q-1} psi[k] / (tau[j+1] - tau[k+1])
        // tau[0] = self.t, tau[j+1] = self.t_hist[j]
        let mut alpha = [T::zero(); MAX_ORDER];
        for j in 0..q {
            let tau_j = if j == 0 {
                self.t
            } else {
                self.t_hist[j - 1]
            };
            let mut lj_prime = -T::one() / psi[j];
            for k in 0..q {
                if k != j {
                    let tau_k = if k == 0 {
                        self.t
                    } else {
                        self.t_hist[k - 1]
                    };
                    lj_prime = lj_prime * psi[k] / (tau_j - tau_k);
                }
            }
            alpha[j] = -lj_prime / l0_prime;
        }

        (gamma, alpha, q)
    }

    fn effective_order(&self) -> usize {
        self.order.min(self.n_hist + 1).min(self.max_order)
    }

    fn predict(&self, t_next: T, _q: usize) -> Y {
        if self.n_hist == 0 {
            return self.y + self.dydt * (t_next - self.t);
        }

        // Use ALL available points for the best possible extrapolation
        let n_pts = self.n_hist + 1;
        let n = self.y.len();
        let mut result = Y::zeros();

        for j in 0..n_pts {
            let t_j = if j == 0 {
                self.t
            } else {
                self.t_hist[j - 1]
            };
            let y_j = if j == 0 { self.y } else { self.y_hist[j - 1] };

            let mut basis = T::one();
            for k in 0..n_pts {
                if k != j {
                    let t_k = if k == 0 {
                        self.t
                    } else {
                        self.t_hist[k - 1]
                    };
                    basis = basis * (t_next - t_k) / (t_j - t_k);
                }
            }
            for i in 0..n {
                result.set(i, result.get(i) + y_j.get(i) * basis);
            }
        }

        result
    }

    fn compute_error_norm(&self, y_corrected: &Y, y_predicted: &Y) -> T {
        let n = y_corrected.len();
        let mut err_sq = T::zero();
        for i in 0..n {
            let sc = self.atol[i] + self.rtol[i] * y_corrected.get(i).abs();
            let e = (y_corrected.get(i) - y_predicted.get(i)) / sc;
            err_sq += e * e;
        }
        (err_sq / T::from_usize(n).unwrap()).sqrt().max(T::from_f64(1e-10).unwrap())
    }

    fn update_history(&mut self) {
        // Shift history: oldest entry is at the end, newest at index 0
        let q = self.effective_order();
        let shift = q.min(self.n_hist).min(MAX_ORDER - 1);
        for i in (1..=shift).rev() {
            self.t_hist[i] = self.t_hist[i - 1];
            self.y_hist[i] = self.y_hist[i - 1];
        }
        self.t_hist[0] = self.t;
        self.y_hist[0] = self.y;
        if self.n_hist < MAX_ORDER {
            self.n_hist += 1;
        }
    }

    fn select_order(&mut self, err: T) {
        let q = self.order;

        if self.reject {
            if q > 1 {
                self.order = q - 1;
            }
            self.steps_at_order = 0;
            return;
        }

        self.steps_at_order += 1;

        // Consider increasing order after enough successful steps
        if self.steps_at_order >= q + 2 && q < self.max_order && self.n_hist >= q {
            // Estimate if order q+1 would give a smaller error
            // Heuristic: if current error is well below tolerance, try higher order
            let threshold = T::from_f64(0.5).unwrap();
            if err < threshold {
                self.order = q + 1;
                self.steps_at_order = 0;
            }
        }
    }
}

impl<T: Real, Y: State<T>> OrdinaryNumericalMethod<T, Y> for Bdf<Ordinary, T, Y> {
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y>,
    {
        let mut evals = Evals::new();

        // Compute initial step size if not provided
        if self.h0 == T::zero() {
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode,
                t0,
                tf,
                y0,
                1,
                &self.rtol,
                &self.atol,
                self.h_min,
                self.h_max,
                &mut evals,
            );
        }

        match validate_step_size_parameters(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h) => self.h = h,
            Err(e) => return Err(e),
        }

        self.t = t0;
        self.y = *y0;
        self.n_hist = 0;
        self.order = 1;
        self.steps_at_order = 0;
        self.steps = 0;
        self.reject = false;
        self.err_old = T::from_f64(1e-4).unwrap();
        self.jacobian_age = usize::MAX;

        ode.diff(t0, y0, &mut self.dydt);
        evals.function += 1;

        let dim = y0.len();
        self.jacobian = Matrix::zeros(dim, dim);
        self.newton_matrix = Matrix::zeros(dim, dim);
        self.rhs = vec![T::zero(); dim];
        self.ip = vec![0; dim];

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
            let e = Error::MaxSteps {
                t: self.t,
                y: self.y,
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        let t_next = self.t + self.h;

        // Compute BDF coefficients for current step sizes and order
        let (gamma, alpha, q) = self.compute_bdf_coefficients(t_next);
        if q == 0 {
            let e = Error::BadInput {
                msg: format!(
                    "BDF order is zero (order={}, n_hist={}, h={}, t={}, t_next={}). Check your problem definition, dimensions, and solver options.",
                    self.order, self.n_hist, self.h, self.t, t_next
                ),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // Predictor: extrapolate from history
        let y_predicted = self.predict(t_next, q);
        let mut y_next = y_predicted;

        // Newton iteration
        let mut newton_converged = false;
        let mut _newt_iter = 0;

        for iter in 0..self.max_newton_iter {
            _newt_iter = iter + 1;

            ode.diff(t_next, &y_next, &mut self.dydt);
            evals.function += 1;

            // Recompute Jacobian and LU decomposition if stale
            if self.jacobian_age > 10 || iter == 0 {
                ode.jacobian(t_next, &y_next, &mut self.jacobian);
                evals.jacobian += 1;
                self.jacobian_age = 0;

                // Newton matrix: M = I - h * gamma * J
                let hg = self.h * gamma;
                for i in 0..dim {
                    for j in 0..dim {
                        self.newton_matrix[(i, j)] = if i == j {
                            T::one() - hg * self.jacobian[(i, j)]
                        } else {
                            -hg * self.jacobian[(i, j)]
                        };
                    }
                }

                if let Err(_) = lu_decomp(&mut self.newton_matrix, &mut self.ip) {
                    self.h = constrain_step_size(
                        self.h * T::from_f64(0.5).unwrap(),
                        self.h_min,
                        self.h_max,
                    );
                    self.h = (self.filter)(self.h);
                    self.status = Status::RejectedStep;
                    self.reject = true;
                    return Ok(evals);
                }
            }

            // Residual: F = y_next - sum(alpha[j] * y_history[j]) - h * gamma * dydt
            let hg = self.h * gamma;
            let mut residual = y_next - self.dydt * hg;

            // Subtract history terms
            for j in 0..q {
                let y_j = if j == 0 { self.y } else { self.y_hist[j - 1] };
                residual = residual - y_j * alpha[j];
            }

            // Solve M * delta = -residual
            for i in 0..dim {
                self.rhs[i] = -residual.get(i);
            }
            if let Err(_) = self.newton_matrix.lin_solve_mut(&mut self.rhs) {
                self.h = constrain_step_size(
                    self.h * T::from_f64(0.5).unwrap(),
                    self.h_min,
                    self.h_max,
                );
                self.h = (self.filter)(self.h);
                self.status = Status::RejectedStep;
                self.reject = true;
                return Ok(evals);
            }

            // Update y_next
            let mut delta_norm = T::zero();
            let mut y_norm = T::zero();
            for i in 0..dim {
                y_next.set(i, y_next.get(i) + self.rhs[i]);
                delta_norm += self.rhs[i] * self.rhs[i];
                y_norm += y_next.get(i) * y_next.get(i);
            }
            delta_norm = delta_norm.sqrt();
            y_norm = y_norm.sqrt();

            if delta_norm < self.newton_tol * (y_norm + T::one()) {
                newton_converged = true;
                break;
            }
        }

        if !newton_converged {
            self.h = constrain_step_size(
                self.h * T::from_f64(0.5).unwrap(),
                self.h_min,
                self.h_max,
            );
            self.h = (self.filter)(self.h);
            self.status = Status::RejectedStep;
            self.reject = true;
            return Ok(evals);
        }

        self.jacobian_age += 1;

        // Error estimation
        let err = self.compute_error_norm(&y_next, &y_predicted);

        if err >= T::one() {
            // Reject step
            let fac = self
                .min_scale
                .max(err.powf(T::one() / T::from_usize(q + 1).unwrap()) / self.safety_factor);
            self.h = constrain_step_size(self.h / fac, self.h_min, self.h_max);
            self.h = (self.filter)(self.h);
            self.status = Status::RejectedStep;
            self.reject = true;
            self.select_order(err);
            return Ok(evals);
        }

        // Accept step
        self.update_history();
        self.t = t_next;
        self.y = y_next;
        self.steps += 1;

        // Step size adjustment
        let fac = self
            .max_scale
            .min(self.min_scale.max(self.safety_factor / err.powf(T::one() / T::from_usize(q + 1).unwrap())));
        let h_new = constrain_step_size(self.h * fac, self.h_min, self.h_max);
        self.h = (self.filter)(h_new);

        self.err_old = err;
        self.reject = false;
        self.select_order(err);

        Ok(evals)
    }

    fn t(&self) -> T {
        self.t
    }

    fn y(&self) -> &Y {
        &self.y
    }

    fn t_prev(&self) -> T {
        if self.n_hist > 0 {
            self.t_hist[0]
        } else {
            self.t
        }
    }

    fn y_prev(&self) -> &Y {
        if self.n_hist > 0 {
            &self.y_hist[0]
        } else {
            &self.y
        }
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

impl<T: Real, Y: State<T>> Interpolation<T, Y> for Bdf<Ordinary, T, Y> {
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        let t0 = self.t_prev();
        let t1 = self.t();

        if t_interp < t0 || t_interp > t1 {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: t0,
                t_curr: t1,
            });
        }

        // Polynomial interpolation through history + current point
        let n_pts = self.n_hist + 1;
        let n = self.y.len();
        let mut result = Y::zeros();

        for j in 0..n_pts {
            let t_j = if j == 0 {
                self.t
            } else {
                self.t_hist[j - 1]
            };
            let y_j = if j == 0 {
                self.y
            } else {
                self.y_hist[j - 1]
            };

            let mut basis = T::one();
            for k in 0..n_pts {
                if k != j {
                    let t_k = if k == 0 {
                        self.t
                    } else {
                        self.t_hist[k - 1]
                    };
                    basis = basis * (t_interp - t_k) / (t_j - t_k);
                }
            }
            for i in 0..n {
                result.set(i, result.get(i) + y_j.get(i) * basis);
            }
        }

        Ok(result)
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
    fn test_bdf_exponential_decay() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf = Bdf::builder().max_order(5);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0])
            .method(bdf)
            .solve()
            .unwrap();
        let actual_val = results.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (actual_val - expected).abs() < 1e-4,
            "BDF exponential decay: got {}, expected {}",
            actual_val,
            expected
        );
    }

    #[test]
    fn test_bdf_order_1() {
        let system = ExponentialDecay { k: 1.0 };
        let bdf = Bdf::builder().max_order(1);
        let results = Ivp::ode(&system, 0.0, 1.0, vector![1.0])
            .method(bdf)
            .solve()
            .unwrap();
        let actual_val = results.y.last().unwrap()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (actual_val - expected).abs() < 1e-2,
            "BDF1 exponential decay: got {}, expected {}",
            actual_val,
            expected
        );
    }

    #[test]
    fn test_bdf_coefficients_order_1() {
        let bdf: Bdf<Ordinary, f64, SVector<f64, 1>> = Bdf::builder();
        let (gamma, alpha, q) = bdf.compute_bdf_coefficients(0.1);
        assert_eq!(q, 1);
        assert!((gamma - 1.0).abs() < 1e-12, "gamma = {}", gamma);
        assert!((alpha[0] - 1.0).abs() < 1e-12, "alpha[0] = {}", alpha[0]);
    }

    #[test]
    fn test_bdf_coefficients_order_2() {
        let mut bdf: Bdf<Ordinary, f64, SVector<f64, 1>> = Bdf::builder();
        bdf.order = 2;
        bdf.n_hist = 1;
        bdf.t = 0.1;
        bdf.t_hist[0] = 0.0;
        let (gamma, alpha, q) = bdf.compute_bdf_coefficients(0.2);
        assert_eq!(q, 2);
        assert!((gamma - 2.0 / 3.0).abs() < 1e-12, "gamma = {}", gamma);
        assert!((alpha[0] - 4.0 / 3.0).abs() < 1e-12, "alpha[0] = {}", alpha[0]);
        assert!(
            (alpha[1] - (-1.0 / 3.0)).abs() < 1e-12,
            "alpha[1] = {}",
            alpha[1]
        );
    }

    #[test]
    fn test_bdf_variable_step_coefficients() {
        let mut bdf: Bdf<Ordinary, f64, SVector<f64, 1>> = Bdf::builder();
        bdf.order = 2;
        bdf.n_hist = 1;
        bdf.t = 0.15;
        bdf.t_hist[0] = 0.0;
        // Step from t=0.15 to t=0.3 (step h=0.15, previous step was 0.15)
        let (gamma, alpha, q) = bdf.compute_bdf_coefficients(0.3);
        assert_eq!(q, 2);
        // With equal step sizes (h_prev = h = 0.15), should match standard BDF2
        assert!((gamma - 2.0 / 3.0).abs() < 1e-12, "gamma = {}", gamma);
        assert!((alpha[0] - 4.0 / 3.0).abs() < 1e-12, "alpha[0] = {}", alpha[0]);
    }
}
