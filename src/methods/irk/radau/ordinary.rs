//! Radau5: 3-stage, order-5 Radau IIA implicit Runge–Kutta with adaptive steps and dense output.

use crate::{
    error::Error,
    linalg::{Matrix, linear, lu},
    methods::{Ordinary, h_init::InitialStepSize, irk::radau::Radau5},
    ode::{ODE, OrdinaryNumericalMethod},
    stats::Evals,
    status::Status,
    traits::{CallBackData, Real, State},
    utils::{constrain_step_size, validate_step_size_parameters},
};

impl<T: Real, Y: State<T>, D: CallBackData> OrdinaryNumericalMethod<T, Y, D>
    for Radau5<Ordinary, T, Y, D>
{
    fn init<F>(&mut self, ode: &F, t0: T, tf: T, y0: &Y) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Initial step size
        if self.h0 == T::zero() {
            // Use ode-specific heuristic that respects the mass matrix
            self.h0 = InitialStepSize::<Ordinary>::compute(
                ode, t0, tf, y0, 5, self.rtol[0], self.atol[0], self.h_min, self.h_max, &mut evals,
            );
        }

        // Validate h0 and align sign with tf direction
        match validate_step_size_parameters::<T, Y, D>(self.h0, self.h_min, self.h_max, t0, tf) {
            Ok(h0) => {
                self.h = h0;
            }
            Err(status) => return Err(status),
        }

        // Delegate to new initializer
        self.initialize(t0, tf, y0)?;

        // ODE uses an identity mass matrix
        let n = y0.len();
        self.mass = Matrix::identity(n);

        ode.diff(self.t, &self.y, &mut self.dydt);
        evals.function += 1;

        // Update previous saved derivative
        self.dydt_prev = self.dydt;

        Ok(evals)
    }

    fn step<F>(&mut self, ode: &F) -> Result<Evals, Error<T, Y>>
    where
        F: ODE<T, Y, D>,
    {
        let mut evals = Evals::new();

        // Number of equations
        let n = self.y.len();

        // Computation of the jacobian
        if self.call_jac {
            evals.jacobian += 1;
            ode.jacobian(self.t, &self.y, &mut self.jacobian);
            self.call_jac = false;
        }

        // Compute the matrices E1 and E2 and their decompositions
        if self.call_decomp {
            let fac1 = self.u1 / self.h;
            let alphn = self.alph / self.h;
            let betan = self.beta / self.h;

            // E1 = -J + fac1*M and LU decompose
            for j in 0..n {
                for i in 0..n {
                    self.e1[(i, j)] = self.mass[(i, j)] * fac1 - self.jacobian[(i, j)];
                }
            }
            if lu::dec(&mut self.e1, &mut self.ip1).is_err() {
                self.singular_count += 1;
                if self.singular_count > 5 {
                    self.status = Status::Error(Error::LinearAlgebra {
                        msg: "Repeated singular matrix in step rejection; aborting.".to_string(),
                    });
                    return Err(Error::LinearAlgebra {
                        msg: "Repeated singular matrix in step rejection; aborting.".to_string(),
                    });
                }
                self.unexpected_step_rejection();
                return Ok(evals);
            }

            // E2R = -J + alphn*M; E2I = betan*M and complex LU decompose
            for j in 0..n {
                for i in 0..n {
                    let m = self.mass[(i, j)];
                    self.e2r[(i, j)] = m * alphn - self.jacobian[(i, j)];
                    self.e2i[(i, j)] = m * betan;
                }
            }
            if lu::decc(&mut self.e2r, &mut self.e2i, &mut self.ip2).is_err() {
                self.singular_count += 1;
                if self.singular_count > 5 {
                    self.status = Status::Error(Error::LinearAlgebra {
                        msg: "Repeated singular matrix in step rejection; aborting.".to_string(),
                    });
                    return Err(Error::LinearAlgebra {
                        msg: "Repeated singular matrix in step rejection; aborting.".to_string(),
                    });
                }
                self.unexpected_step_rejection();
                return Ok(evals);
            }
            evals.decompositions += 1;
        }

        // Main step begins
        self.steps += 1;

        // Max steps guard
        if self.steps >= self.max_steps {
            self.status = Status::Error(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
            return Err(Error::MaxSteps {
                t: self.t,
                y: self.y,
            });
        }

        // Step size guard
        if self.h.abs() < self.h_prev.abs() * self.uround {
            self.status = Status::Error(Error::StepSize {
                t: self.t,
                y: self.y,
            });
            return Err(Error::StepSize {
                t: self.t,
                y: self.y,
            });
        }

        // Index-2 scaling: scal[i] /= hhfac for index-2 variables
        for &i in &self.index2 {
            let val = self.scal.get(i) / self.hhfac;
            self.scal.set(i, val);
        }
        // Index-3 scaling: scal[i] /= (hhfac * hhfac) for index-3 variables
        for &i in &self.index3 {
            let val = self.scal.get(i) / (self.hhfac * self.hhfac);
            self.scal.set(i, val);
        }

        // Starting values for Newton iteration
        if self.first {
            for i in 0..3 {
                self.z[i] = Y::zeros();
                self.f[i] = Y::zeros();
            }
        } else {
            let c3q = self.h / self.h_prev;
            let c1q = self.c1 * c3q;
            let c2q = self.c2 * c3q;

            let ak1 = self.cont[1];
            let ak2 = self.cont[2];
            let ak3 = self.cont[3];

            self.z[0] = (ak1 + (ak2 + ak3 * (c1q - self.c1m1)) * (c1q - self.c2m1)) * c1q;
            self.z[1] = (ak1 + (ak2 + ak3 * (c2q - self.c1m1)) * (c2q - self.c2m1)) * c2q;
            self.z[2] = (ak1 + (ak2 + ak3 * (c3q - self.c1m1)) * (c3q - self.c2m1)) * c3q;
            
            self.f[0] = self.z[0] * self.tinv[(0, 0)] + self.z[1] * self.tinv[(0, 1)] + self.z[2] * self.tinv[(0, 2)];
            self.f[1] = self.z[0] * self.tinv[(1, 0)] + self.z[1] * self.tinv[(1, 1)] + self.z[2] * self.tinv[(1, 2)];
            self.f[2] = self.z[0] * self.tinv[(2, 0)] + self.z[1] * self.tinv[(2, 1)] + self.z[2] * self.tinv[(2, 2)];
        }

        // Loop for simplified newton iteration
        self.faccon = self
            .faccon
            .max(self.uround)
            .powf(T::from_f64(0.8).unwrap());
        self.theta = self.thet.abs();
        let mut newt_iter: usize = 0;
        'newton: loop {
            if newt_iter >= self.max_newton_iter {
                self.unexpected_step_rejection();
                return Ok(evals);
            }

            newt_iter += 1;

            // Compute the stages
            let t1 = self.t + self.c1 * self.h;
            let t2 = self.t + self.c2 * self.h;
            let t3 = self.t + self.h;
            let y1 = self.y + self.z[0];
            let y2 = self.y + self.z[1];
            let y3 = self.y + self.z[2];

            ode.diff(t1, &y1, &mut self.k[0]);
            ode.diff(t2, &y2, &mut self.k[1]);
            ode.diff(t3, &y3, &mut self.k[2]);
            evals.function += 3;

            // Solve the linear systems
            self.z[0] = self.k[0] * self.tinv[(0, 0)]
                + self.k[1] * self.tinv[(0, 1)]
                + self.k[2] * self.tinv[(0, 2)];
            self.z[1] = self.k[0] * self.tinv[(1, 0)]
                + self.k[1] * self.tinv[(1, 1)]
                + self.k[2] * self.tinv[(1, 2)];
            self.z[2] = self.k[0] * self.tinv[(2, 0)]
                + self.k[1] * self.tinv[(2, 1)]
                + self.k[2] * self.tinv[(2, 2)];

            let fac1 = self.u1 / self.h;
            let alphn = self.alph / self.h;
            let betan = self.beta / self.h;

            // Assemble RHS and solve for (Z1, Z2, Z3)
            for i in 0..n {
                // S1 = -(M row i)·F1; S2 = -(M row i)·F2; S3 = -(M row i)·F3
                let mut s1 = T::zero();
                let mut s2 = T::zero();
                let mut s3 = T::zero();
                for j in 0..n {
                    let mij = self.mass[(i, j)];
                    s1 = s1 - mij * self.f[0].get(j);
                    s2 = s2 - mij * self.f[1].get(j);
                    s3 = s3 - mij * self.f[2].get(j);
                }
                self.z[0].set(i, self.z[0].get(i) + s1 * fac1);
                self.z[1].set(i, self.z[1].get(i) + s2 * alphn - s3 * betan);
                self.z[2].set(i, self.z[2].get(i) + s3 * alphn + s2 * betan);
            }

            // Solve E1 * Z1 = RHS1 (real system)
            let z1 = &mut self.z[0];
            linear::sol(&self.e1, z1, &self.ip1);

            // Solve complex system for (Z2, Z3)
            let (z12, z3) = self.z.split_at_mut(2);
            let z2 = &mut z12[1];
            let z3 = &mut z3[0];
            linear::solc(&self.e2r, &self.e2i, z2, z3, &self.ip2);

            // Record solves
            evals.solves += 2;
            evals.newton += 1;

            // Convergence control
            let mut dyno = T::zero();
            for i in 0..n {
                let sc = self.scal.get(i);
                let v1 = self.z[0].get(i) / sc;
                let v2 = self.z[1].get(i) / sc;
                let v3 = self.z[2].get(i) / sc;
                dyno = dyno + v1 * v1 + v2 * v2 + v3 * v3;
            }
            dyno = (dyno / T::from_f64((3 * n) as f64).unwrap()).sqrt();

            // Bad convergence or number of iterations is too large
            if newt_iter > 1 && newt_iter < self.max_newton_iter {
                let thq = dyno / self.dynold;
                if newt_iter == 2 {
                    self.theta = thq;
                } else {
                    self.theta = (thq * self.thqold).sqrt();
                }
                self.thqold = thq;

                if self.theta < T::from_f64(0.99).unwrap() {
                    self.faccon = self.theta / (T::one() - self.theta);
                    let remaining_iters = (self.max_newton_iter - 1 - newt_iter) as f64;
                    let dyth =
                        self.faccon * dyno * self.theta.powf(T::from_f64(remaining_iters).unwrap())
                            / self.newton_tol;
                    if dyth >= T::one() {
                        let qnewt = T::from_f64(1e-4)
                            .unwrap()
                            .max(T::from_f64(20.0).unwrap().min(dyth));
                        let exponent = -T::one() / T::from_f64(4.0 + remaining_iters).unwrap();
                        self.hhfac = T::from_f64(0.8).unwrap() * qnewt.powf(exponent);
                        self.h = self.h * self.hhfac;
                        self.status = Status::RejectedStep;
                        self.reject = true;
                        return Ok(evals);
                    }
                } else {
                    self.unexpected_step_rejection();
                    return Ok(evals);
                }
            }
            self.dynold = dyno.max(self.uround);

            // Compute new F and Z
            self.f[0] = self.f[0] + self.z[0];
            self.f[1] = self.f[1] + self.z[1];
            self.f[2] = self.f[2] + self.z[2];

            self.z[0] = self.f[0] * self.tmat[(0, 0)]
                + self.f[1] * self.tmat[(0, 1)]
                + self.f[2] * self.tmat[(0, 2)];
            self.z[1] = self.f[0] * self.tmat[(1, 0)]
                + self.f[1] * self.tmat[(1, 1)]
                + self.f[2] * self.tmat[(1, 2)];
            self.z[2] = self.f[0] * self.tmat[(2, 0)] + self.f[1];

            // Check Newton tolerance
            if self.faccon * dyno > self.newton_tol {
                continue 'newton;
            } else {
                break 'newton;
            }
        }

        // Error estimation
        let hee1 = self.dd1 / self.h;
        let hee2 = self.dd2 / self.h;
        let hee3 = self.dd3 / self.h;
        let mut f1 = self.z[0] * hee1 + self.z[1] * hee2 + self.z[2] * hee3;
        let mut f2 = Y::zeros();
        let mut cont = Y::zeros();
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + self.mass[(i, j)] * f1.get(j);
            }
            f2.set(i, sum);
            cont.set(i, sum + self.dydt.get(i));
        }
        linear::sol(&self.e1, &mut cont, &self.ip1);
        evals.solves += 1;

        // Error estimate
        let mut err = T::zero();
        for i in 0..n {
            let r = cont.get(i) / self.scal.get(i);
            err = err + r * r;
        }
        let mut err = (err / T::from_usize(n).unwrap())
            .sqrt()
            .max(T::from_f64(1e-10).unwrap());

        // Optional refinement: on first or rejected step and large error
        if err >= T::one() && (self.first || self.reject) {
            cont = self.y + cont;
            f1 = Y::zeros();
            ode.diff(self.t, &cont, &mut f1);
            evals.function += 1;

            // cont = f1 + F2; solve again
            cont = f1 + f2;
            linear::sol(&self.e1, &mut cont, &self.ip1);
            evals.solves += 1;

            // Recompute error
            err = T::zero();
            for i in 0..n {
                let r = cont.get(i) / self.scal.get(i);
                err = err + r * r;
            }
            err = (err / T::from_usize(n).unwrap())
                .sqrt()
                .max(T::from_f64(1e-10).unwrap());
        }

        // Computation of hnew (configurable controller parameters)
        let fac = self.safety_factor.min(
            self.cfac
                / (T::from_usize(newt_iter).unwrap()
                    + T::from_f64(2.0).unwrap() * T::from_usize(self.max_newton_iter).unwrap()),
        );
        let mut quot = self
            .facr
            .max(self.facl.min(err.powf(T::from_f64(0.25).unwrap()) / fac));
        let mut hnew = self.h / quot;

        // Is the error small enough?
        if err < T::one() {
            // Step accepted
            self.first = false;
            self.n_accepted += 1;

            // Predicttive Gustafsson controller
            if self.predictive {
                if self.n_accepted > 1 {
                    let mut facgus = (self.h_acc / self.h)
                        * (err * err / self.err_acc).powf(T::from_f64(0.25).unwrap())
                        / self.safety_factor;
                    facgus = self.facr.max(self.facl.min(facgus));
                    quot = quot.max(facgus);
                    hnew = self.h / quot;
                }
                self.h_acc = self.h;
                self.err_acc = err.max(T::from_f64(1e-2).unwrap());
            }

            // Store previous values for interpolation
            self.t_prev = self.t;
            self.y_prev = self.y;
            self.dydt_prev = self.dydt;
            self.h_prev = self.h;

            // y_{n+1} = y_n + z₃; t_{n+1} = t_n + h
            self.y = self.y + self.z[2];
            self.t = self.t + self.h;

            // New derivative at (y_{n+1}, t_{n+1})
            ode.diff(self.t, &self.y, &mut self.dydt);
            evals.function += 1;

            // Dense output coefficients
            let z1 = self.z[0];
            let z2 = self.z[1];
            let z3 = self.z[2];

            let c1 = (z2 - z3) / self.c2m1;
            let ak = (z1 - z2) / self.c1mc2;
            let mut acont3 = z1 / self.c1;
            acont3 = (ak - acont3) / self.c2;
            let c2 = (ak - c1) / self.c1m1;
            let c3 = c2 - acont3;

            self.cont = [Y::zeros(); 4];
            self.cont[0] = self.y;
            self.cont[1] = c1;
            self.cont[2] = c2;
            self.cont[3] = c3;

            // Compute error scale
            for i in 0..n {
                self.scal
                    .set(i, self.atol[i] + self.rtol[i] * self.y.get(i).abs());
            }

            // Constrain new step size to [h_min, h_max]
            hnew = constrain_step_size(hnew, self.h_min, self.h_max);

            // Prevent oscillations due to previous step rejection
            if self.reject {
                let posneg = self.h.signum();
                hnew = posneg * hnew.abs().min(self.h.abs());
                self.reject = false;
                self.status = Status::Solving;
            }

            // Sophisticated step size control
            let qt = hnew / self.h;
            self.hhfac = self.h;
            if self.theta < self.thet && qt > self.quot1 && qt < self.quot2 {
                // Skip jacobian and decomposition recomputation
                self.call_decomp = false;
                self.call_jac = false;
                return Ok(evals);
            };
            self.h = hnew;
            self.hhfac = self.h;

            if self.theta < self.thet {
                // Skip jacobian recomputation
                self.call_jac = false;
                return Ok(evals);
            }

            // Next step does everything.
            self.call_jac = true;
            self.call_decomp = true;
        } else {
            // Step rejected
            self.reject = true;
            self.status = Status::RejectedStep;

            // If first step, reduce more aggressively
            if self.first {
                self.h = self.h * T::from_f64(0.1).unwrap();
                self.hhfac = T::from_f64(0.1).unwrap();
            } else {
                self.hhfac = hnew / self.h;
                self.h = hnew;
            }

            // Constrain step size to [h_min, h_max]
            self.h = constrain_step_size(self.h, self.h_min, self.h_max);
        }

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
        self.h = h;
    }
    fn status(&self) -> &Status<T, Y, D> {
        &self.status
    }
    fn set_status(&mut self, status: Status<T, Y, D>) {
        self.status = status;
    }
}
