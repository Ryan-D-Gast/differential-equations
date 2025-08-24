use crate::error::Error;
use crate::linalg::Matrix;
use crate::methods::irk::radau::Radau5;
use crate::status::Status;
use crate::traits::{CallBackData, Real, State};

impl<E, T: Real, Y: State<T>, D: CallBackData> Radau5<E, T, Y, D> {
    /// Initialize Radau5: combines `set_parameters` and common workspace setup.
    pub fn initialize(&mut self, t0: T, tf: T, y0: &Y) -> Result<(), Error<T, Y>> {
        // Dimension of system
        let n = y0.len();

        // UROUND: if user provided nonsense, flag error; else keep default 1e-16
        if self.uround <= T::from_f64(1e-19).unwrap() || self.uround >= T::one() {
            let e = Error::BadInput {
                msg: "UROUND is out of range (expected ~1e-16).".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // Tolerances
        let ten = T::from_f64(10.0).unwrap();
        if self.atol[0] <= T::zero() || self.rtol[0] <= ten * self.uround {
            let e = Error::BadInput {
                msg: "Tolerances are too small (require ATOL > 0 and RTOL > 10*UROUND)."
                    .to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }
        // Preserve ratio and scale
        for i in 0..n {
            let quot = self.atol[i] / self.rtol[i];
            let expm = T::from_f64(2.0 / 3.0).unwrap();
            self.rtol[i] = T::from_f64(0.1).unwrap() * self.rtol[i].powf(expm);
            self.atol[i] = self.rtol[i] * quot;
        }

        // NMAX must be positive
        if self.max_steps == 0 {
            let e = Error::BadInput {
                msg: "max_steps must be > 0".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // NIT must be positive
        if self.max_newton_iter == 0 {
            let e = Error::BadInput {
                msg: "max_newton_iter must be > 0".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // SAFE in (0.001, 1.0)
        if !(self.safety_factor > T::from_f64(0.001).unwrap() && self.safety_factor < T::one()) {
            let e = Error::BadInput {
                msg: "safety_factor must be in (0.001, 1.0)".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // THET < 1.0
        if self.thet >= T::one() {
            let e = Error::BadInput {
                msg: "thet must be < 1.0".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // FNEWT: compute default if zero, else validate lower bound
        let tolst = self.rtol[0]; // after adjust_tolerances
        if self.newton_tol <= T::zero() {
            let upper = T::from_f64(0.03).unwrap().min(tolst.sqrt());
            let lower = T::from_f64(10.0).unwrap() * self.uround / tolst;
            self.newton_tol = lower.max(upper);
        } else {
            let min_allowed = self.uround / tolst;
            if self.newton_tol <= min_allowed {
                let e = Error::BadInput {
                    msg: "newton_tol too small (<= UROUND/RTOL')".to_string(),
                };
                self.status = Status::Error(e.clone());
                return Err(e);
            }
        }

        // QUOT window: require quot1 <= 1 and quot2 >= 1
        if self.quot1 > T::one() || self.quot2 < T::one() {
            let e = Error::BadInput {
                msg: "Invalid (quot1, quot2): require quot1 <= 1 and quot2 >= 1".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // HMAX default: if unset or infinite, use |tf - t0|
        if !self.h_max.is_finite() || self.h_max <= T::zero() {
            self.h_max = (tf - t0).abs();
        }

        // Clamp factors: reconcile user-facing min/max scales with facl/facr
        // facl = 1/min_scale (>= 1), facr = 1/max_scale (<= 1)
        if self.min_scale <= T::zero() || self.min_scale > T::one() {
            let e = Error::BadInput {
                msg: "min_scale must be in (0, 1] (default 0.2)".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }
        if self.max_scale < T::one() {
            let e = Error::BadInput {
                msg: "max_scale must be >= 1 (default 8.0)".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }
        self.facl = T::one() / self.min_scale;
        self.facr = T::one() / self.max_scale;
        if self.facl < T::one() || self.facr > T::one() {
            let e = Error::BadInput {
                msg: "Invalid clamp factors derived from scales (facl>=1, facr<=1)".to_string(),
            };
            self.status = Status::Error(e.clone());
            return Err(e);
        }

        // Composite safety factor used in step-size prediction
        self.cfac = self.safety_factor
            * (T::one() + T::from_f64(2.0).unwrap() * T::from_usize(self.max_newton_iter).unwrap());

        // Reset stats
        self.steps = 0;
        self.rejects = 0;
        self.n_accepted = 0;
        self.jacobian_age = 0;

        // Initialize state
        self.t = t0;
        self.tf = tf;
        self.y = *y0;

        // Size the mass matrix as identity (DE/DAE specific code should overwrite)
        self.mass = Matrix::identity(n);

        // Do not call diff here; leave dydt zeroed
        self.dydt = Y::zeros();

        // Previous state
        self.t_prev = self.t;
        self.y_prev = self.y;
        self.dydt_prev = self.dydt;
        self.h_prev = self.h;

        // Step size factor
        self.hhfac = self.h;

        // Calculate tolerance
        for i in 0..n {
            self.scal
                .set(i, self.atol[i] + self.rtol[i] * self.y.get(i).abs());
        }

        // Workspace
        self.z = [Y::zeros(), Y::zeros(), Y::zeros()];
        self.k = [Y::zeros(), Y::zeros(), Y::zeros()];
        self.f = [Y::zeros(), Y::zeros(), Y::zeros()];
        self.jacobian = Matrix::zeros(n, n);
        self.e1 = Matrix::zeros(n, n);
        self.e2r = Matrix::zeros(n, n);
        self.e2i = Matrix::zeros(n, n);
        self.ip1 = vec![0; n];
        self.ip2 = vec![0; n];
        self.a = Matrix::zeros(2 * n, 2 * n);
        self.b = vec![T::zero(); 2 * n];

        // Dense output coefficients
        self.cont = [Y::zeros(); 4];

        // Flags
        self.first = true;
        self.reject = false;

        self.status = Status::Initialized;
        Ok(())
    }
}
