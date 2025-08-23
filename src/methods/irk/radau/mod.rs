//! IRK with Newton solves and adaptive step size.

mod algebraic;
mod initialize;
mod interpolate;
mod ordinary;

use std::marker::PhantomData;

use crate::{
    linalg::Matrix,
    methods::{ImplicitRungeKutta, Radau},
    status::Status,
    traits::{CallBackData, Real, State},
    utils::constrain_step_size,
};

/// Constructor for Radau5
impl<E, T: Real, Y: State<T>, D: CallBackData> ImplicitRungeKutta<E, Radau, T, Y, D, 5, 3, 3> {
    /// Creates a new Radau IIA 3-stage implicit Runge-Kutta method of order 5.
    pub fn radau5() -> Radau5<E, T, Y, D> {
        Radau5::default()
    }
}

/// Radau IIA(5) implicit RK with adaptive step size and dense output.
pub struct Radau5<E, T: Real, Y: State<T>, D: CallBackData> {
    // Configuration
    /// Relative error tolerance for adaptive step size control
    pub rtol: T,
    /// Absolute error tolerance for adaptive step size control  
    pub atol: T,
    /// Initial step size (computed automatically if zero)
    pub h0: T,
    /// Minimum allowed step size
    pub h_min: T,
    /// Maximum allowed step size
    pub h_max: T,
    /// Maximum number of integration steps
    pub max_steps: usize,
    /// Maximum number of consecutive step rejections
    pub max_rejects: usize,
    /// Newton iteration convergence tolerance
    pub newton_tol: T,
    /// Maximum Newton iterations per step
    pub max_newton_iter: usize,
    /// Safety factor for step size control (typically 0.8-0.9)
    pub safety_factor: T,
    /// Minimum step size scaling factor
    pub min_scale: T,
    /// Maximum step size scaling factor
    pub max_scale: T,
    /// Enable predictive (Gustafsson) step-size controller
    pub predictive: bool,

    // State
    /// Current time
    t: T,
    /// Current solution vector
    y: Y,
    /// Current derivative vector dy/dt
    dydt: Y,
    /// Previous time (for interpolation)
    t_prev: T,
    /// Previous solution vector
    y_prev: Y,
    /// Previous derivative vector
    dydt_prev: Y,
    /// Current step size
    h: T,
    /// Previous step size
    h_prev: T,
    /// Final integration time
    tf: T,

    // Method constants
    /// First collocation point: c₁ = (4-√6)/10
    c1: T,
    /// Second collocation point: c₂ = (4+√6)/10
    c2: T,
    /// Helper constant: c₁ - 1
    c1m1: T,
    /// Helper constant: c₂ - 1
    c2m1: T,
    /// Helper constant: c₁ - c₂
    c1mc2: T,
    /// Error estimation coefficient 1
    dd1: T,
    /// Error estimation coefficient 2
    dd2: T,
    /// Error estimation coefficient 3
    dd3: T,
    /// Real system coefficient: 1/u₁
    u1: T,
    /// Complex system real part coefficient
    alph: T,
    /// Complex system imaginary part coefficient
    beta: T,

    /// Transformation matrix T (3×3) for stage variables
    tmat: Matrix<T>,
    /// Inverse transformation matrix T⁻¹ (3×3)
    tinv: Matrix<T>,

    // Workspace
    /// Stage solution vectors: y + Zᵢ at collocation points
    z: [Y; 3],
    /// Stage derivative estimates
    k: [Y; 3],
    /// Right-hand side evaluations at stages
    f: [Y; 3],
    /// Jacobian matrix ∂f/∂y
    jacobian: Matrix<T>,
    /// Age of current Jacobian
    jacobian_age: usize,
    /// Solving linear system workspaces
    /// a: n2 x n2 dense matrix, b: length-n2 RHS vector
    a: Matrix<T>,
    b: Vec<T>,

    // Newton convergence control
    /// Newton convergence factor
    faccon: T,
    /// Previous Newton norm
    dynold: T,
    /// Newton convergence rate
    theta: T,
    /// Decides whether the jacobian should be recomputed;
    thet: T,
    /// Previous convergence quotient
    thqold: T,

    // Sophisticated step size control
    /// Composite safety factor used in step-size control: safety_factor * (1 + 2*max_newton_iter)
    /// Users can override this; default is derived from safety_factor and max_newton_iter.
    cfac: T,
    /// Minimum clamp factor for step-size change (e.g., 1/8).
    facr: T,
    /// Maximum clamp factor for step-size change (e.g., 5.0).
    facl: T,
    /// Rounding Unit
    uround: T,
    /// Error scale, scal = atol + rtol * abs(y)
    scal: Y,
    /// quot1 < hnew/hold < quot2
    quot1: T,
    quot2: T,
    /// Factor for new step size
    hhfac: T,

    // Linear system matrices
    /// Real system matrix: (M - h*u₁*J) for first linear system
    e1: Matrix<T>,
    /// Complex system real part: (M - h*α*J)
    e2r: Matrix<T>,
    /// Complex system imaginary part: (-h*β*J)
    e2i: Matrix<T>,

    // Pivot vectors from LU decomposition
    /// Pivot vector for real system E1
    ip1: Vec<usize>,
    /// Pivot vector for complex system E2
    ip2: Vec<usize>,

    /// Mass matrix M(t,y) for DAE systems
    mass: Matrix<T>,
    /// Indexs in state vector at which index two algebraic equations are located
    index2: Vec<usize>,
    /// Indexs in state vector at which index three algebraic equations are located
    index3: Vec<usize>,

    // Statistics
    /// Count of consecutive singular matrix encounters
    singular_count: usize,
    /// Total integration steps taken
    steps: usize,
    /// Total step rejections
    rejects: usize,
    /// Total accepted steps (for Gustafsson's controller)
    n_accepted: usize,
    /// Current solver status
    status: Status<T, Y, D>,

    // Dense output
    /// Continuous output coefficients for dense output polynomial
    cont: [Y; 4],

    // Control flags
    /// True for the first integration step
    first: bool,
    /// True when last step was rejected
    reject: bool,
    /// Routing flag for Jacobian computation
    calc_jac: bool,
    /// Routing flag for Jacobian decomposition
    calc_decomp: bool,

    // Gustafsson controller
    /// Last accepted step size HACC
    h_acc: T,
    /// Last accepted error ERRACC
    err_acc: T,

    /// Equation type
    equation: PhantomData<E>,
}

impl<E, T: Real, Y: State<T>, D: CallBackData> Default for Radau5<E, T, Y, D> {
    fn default() -> Self {
        // Radau IIA(5) constants

        // Collocation points for Radau IIA(5)
        let sqrt6 = 6.0_f64.sqrt();
        let sq6 = T::from_f64(sqrt6).unwrap();
        let c1_t = (T::from_f64(4.0).unwrap() - sq6) / T::from_f64(10.0).unwrap();
        let c2_t = (T::from_f64(4.0).unwrap() + sq6) / T::from_f64(10.0).unwrap();
        let c1m1 = c1_t - T::one();
        let c2m1 = c2_t - T::one();
        let c1mc2 = c1_t - c2_t;

        // Error estimation coefficients
        let dd1 = -(T::from_f64(13.0).unwrap() + T::from_f64(7.0).unwrap() * sq6)
            / T::from_f64(3.0).unwrap();
        let dd2 = (-T::from_f64(13.0).unwrap() + T::from_f64(7.0).unwrap() * sq6)
            / T::from_f64(3.0).unwrap();
        let dd3 = -T::one() / T::from_f64(3.0).unwrap();

        // Coefficients for Newton iteration
        let p81_13 = T::from_f64(81.0_f64.powf(1.0 / 3.0)).unwrap();
        let p9_13 = T::from_f64(9.0_f64.powf(1.0 / 3.0)).unwrap();
        let sqrt3 = T::from_f64(3.0_f64.sqrt()).unwrap();
        let mut u1 = (T::from_f64(6.0).unwrap() + p81_13 - p9_13) / T::from_f64(30.0).unwrap();
        let mut alph = (T::from_f64(12.0).unwrap() - p81_13 + p9_13) / T::from_f64(60.0).unwrap();
        let mut beta = (p81_13 + p9_13) * sqrt3 / T::from_f64(60.0).unwrap();
        let cno = alph * alph + beta * beta;
        u1 = T::one() / u1;
        alph = alph / cno;
        beta = beta / cno;

        // Transformation matrices
        let mut tmat = Matrix::zeros(3, 3);
        tmat[(0, 0)] = T::from_f64(9.1232394870892942792E-02).unwrap();
        tmat[(0, 1)] = T::from_f64(-0.14125529502095420843E0).unwrap();
        tmat[(0, 2)] = T::from_f64(-3.0029194105147424492E-02).unwrap();
        tmat[(1, 0)] = T::from_f64(0.24171793270710701896E0).unwrap();
        tmat[(1, 1)] = T::from_f64(0.20412935229379993199E0).unwrap();
        tmat[(1, 2)] = T::from_f64(0.38294211275726193779E0).unwrap();
        tmat[(2, 0)] = T::from_f64(0.96604818261509293619E0).unwrap();

        let mut tinv = Matrix::zeros(3, 3);
        tinv[(0, 0)] = T::from_f64(4.3255798900631553510E0).unwrap();
        tinv[(0, 1)] = T::from_f64(0.33919925181580986954E0).unwrap();
        tinv[(0, 2)] = T::from_f64(0.54177053993587487119E0).unwrap();
        tinv[(1, 0)] = T::from_f64(-4.1787185915519047273E0).unwrap();
        tinv[(1, 1)] = T::from_f64(-0.32768282076106238708E0).unwrap();
        tinv[(1, 2)] = T::from_f64(0.47662355450055045196E0).unwrap();
        tinv[(2, 0)] = T::from_f64(-0.50287263494578687595E0).unwrap();
        tinv[(2, 1)] = T::from_f64(2.5719269498556054292E0).unwrap();
        tinv[(2, 2)] = T::from_f64(-0.59603920482822492497E0).unwrap();

        // Default step-size controller parameters
        let safety_factor = T::from_f64(0.9).unwrap();
        let max_newton_iter_usize: usize = 7;
        let cfac_default = safety_factor
            * (T::one()
                + T::from_f64(2.0).unwrap() * T::from_usize(max_newton_iter_usize).unwrap());
        let facl_default = T::from_f64(5.0).unwrap();
        let facr_default = T::from_f64(1.0 / 8.0).unwrap();

        // Default tolerances and derived Newton tolerance
        let rtol_default = T::from_f64(1e-6).unwrap();
        let atol_default = T::from_f64(1e-6).unwrap();
        let uround = T::from_f64(1e-16).unwrap();
        let tolst = T::from_f64(0.1).unwrap() * rtol_default.powf(T::from_f64(2.0 / 3.0).unwrap());
        let fn_upper = T::from_f64(0.03).unwrap().min(tolst.sqrt());
        let fn_lower = T::from_f64(10.0).unwrap() * uround / tolst;
        let newton_tol_default = fn_lower.max(fn_upper);

        Self {
            // Settings
            rtol: rtol_default,
            atol: atol_default,
            h0: T::zero(),
            h_min: T::zero(),
            h_max: T::infinity(),
            max_steps: 100_000,
            max_rejects: 20,
            newton_tol: newton_tol_default,
            max_newton_iter: max_newton_iter_usize,
            safety_factor,
            cfac: cfac_default,
            facl: facl_default,
            facr: facr_default,
            min_scale: T::from_f64(0.2).unwrap(),
            max_scale: T::from_f64(8.0).unwrap(),

            // Algorithm toggles
            predictive: true,

            // State
            t: T::zero(),
            y: Y::zeros(),
            dydt: Y::zeros(),
            t_prev: T::zero(),
            y_prev: Y::zeros(),
            dydt_prev: Y::zeros(),
            h: T::zero(),
            h_prev: T::zero(),
            tf: T::zero(),

            // Method constants
            c1: c1_t,
            c2: c2_t,
            c1m1,
            c2m1,
            c1mc2,
            dd1,
            dd2,
            dd3,
            u1,
            alph,
            beta,
            tmat,
            tinv,

            // Workspace
            z: [Y::zeros(); 3],
            k: [Y::zeros(); 3],
            f: [Y::zeros(); 3],
            jacobian: Matrix::zeros(0, 0),
            jacobian_age: 0,
            a: Matrix::zeros(0, 0),
            b: Vec::new(),

            // Newton convergence control
            faccon: T::one(),
            dynold: T::from_f64(1e-16).unwrap(),
            theta: T::zero(),
            thet: T::from_f64(0.001).unwrap(),
            thqold: T::one(),

            // Step size control variables
            uround,
            scal: Y::zeros(),
            quot1: T::one(),
            quot2: T::from_f64(1.2).unwrap(),
            hhfac: T::zero(),

            // Linear system matrices
            e1: Matrix::zeros(0, 0),
            e2r: Matrix::zeros(0, 0),
            e2i: Matrix::zeros(0, 0),

            // Pivot vectors
            ip1: Vec::new(),
            ip2: Vec::new(),

            // Mass matrix
            mass: Matrix::identity(0),
            index2: Vec::new(),
            index3: Vec::new(),

            // Dense output coefficients
            cont: [Y::zeros(); 4],

            // Error recovery
            singular_count: 0,

            // Statistics
            steps: 0,
            rejects: 0,
            n_accepted: 0,
            status: Status::Uninitialized,

            // Control flags
            first: true,
            reject: false,
            calc_jac: true,
            calc_decomp: true,

            // Predictive controller defaults
            h_acc: T::zero(),
            err_acc: T::from_f64(1e-2).unwrap(),

            // Equation type
            equation: PhantomData,
        }
    }
}

impl<E, T: Real, Y: State<T>, D: CallBackData> Radau5<E, T, Y, D> {
    // Builder methods
    pub fn rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }

    pub fn h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    pub fn h_min(mut self, h_min: T) -> Self {
        self.h_min = h_min;
        self
    }

    pub fn h_max(mut self, h_max: T) -> Self {
        self.h_max = h_max;
        self
    }

    pub fn min_scale(mut self, min_scale: T) -> Self {
        self.min_scale = min_scale;
        self.facl = T::one() / min_scale;
        self
    }

    pub fn max_scale(mut self, max_scale: T) -> Self {
        self.max_scale = max_scale;
        self.facr = T::one() / max_scale;
        self
    }

    /// Enable/disable predictive (Gustafsson) step-size controller.
    pub fn predictive(mut self, enabled: bool) -> Self {
        self.predictive = enabled;
        self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    pub fn max_rejects(mut self, n: usize) -> Self {
        self.max_rejects = n;
        self
    }

    pub fn newton_tol(mut self, tol: T) -> Self {
        self.newton_tol = tol;
        self
    }

    pub fn safety_factor(mut self, sf: T) -> Self {
        self.safety_factor = sf;
        self
    }

    pub fn max_newton_iter(mut self, n: usize) -> Self {
        self.max_newton_iter = n;
        self
    }

    /// Indexes in the state vector of the index two algebraic equations
    pub fn index2_equations_idxs(mut self, index2eqs: Vec<usize>) -> Self {
        self.index2 = index2eqs;
        self
    }

    /// Indexes in the state vector of the index three algebraic equations
    pub fn index3_equations_idxs(mut self, index3eqs: Vec<usize>) -> Self {
        self.index3 = index3eqs;
        self
    }

    fn unexpected_step_rejection(&mut self) {
        self.hhfac = T::from_f64(0.5).unwrap();
        self.h = constrain_step_size(self.h * self.hhfac, self.h_min, self.h_max);
        self.reject = true;
        self.status = Status::RejectedStep;
    }
}
