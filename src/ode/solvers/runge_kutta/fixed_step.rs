//! Fixed-step Runge-Kutta methods for solving ordinary differential equations.

use crate::runge_kutta_method;

runge_kutta_method!(
    /// Euler's Method (1st Order Runge-Kutta) for solving ordinary differential equations.
    /// 
    /// Euler's method is the simplest form of Runge-Kutta methods, and is a first-order method also known as RK1.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0 | 0
    /// -----
    ///   | 1
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Euler_method)
    name: Euler,
    a: [[0.0]],
    b: [1.0],
    c: [0.0],
    order: 1,
    stages: 1
);

runge_kutta_method!(
    /// Midpoint Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1/2 | 1/2
    /// ------------
    ///     | 0   1
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Midpoint_method)
    name: Midpoint,
    a: [[0.0, 0.0],
        [0.5, 0.0]],
    b: [0.0, 1.0],
    c: [0.0, 0.5],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Heun's Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1   | 1
    /// ------------
    ///     | 1/2 1/2
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Heun%27s_method)
    name: Heun,
    a: [[0.0, 0.0],
        [1.0, 0.0]],
    b: [0.5, 0.5],
    c: [0.0, 1.0],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Ralston's Method (2nd Order Runge-Kutta) for solving ordinary differential equations.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 2/3 | 2/3
    /// ------------
    ///     | 1/4 3/4
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Second-order_methods_with_two_stages)
    name: Ralston,
    a: [[0.0, 0.0],
        [2.0/3.0, 0.0]],
    b: [1.0/4.0, 3.0/4.0],
    c: [0.0, 2.0/3.0],
    order: 2,
    stages: 2
);

runge_kutta_method!(
    /// Classic Runge-Kutta 4 method for solving ordinary differential equations.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 0.5 | 0.5
    /// 0.5 | 0   0.5
    /// 1   | 0   0   1
    /// ---------------------
    ///    | 1/6 1/3 1/3 1/6
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples)
    name: RK4,
    a: [[0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]],
    b: [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
    c: [0.0, 0.5, 0.5, 1.0],
    order: 4,
    stages: 4
);

runge_kutta_method!(
    /// Three-Eighths Rule (4th Order Runge-Kutta) for solving ordinary differential equations.
    /// The primary advantage this method has is that almost all of the error coefficients 
    /// are smaller than in the popular method, but it requires slightly more FLOPs 
    /// (floating-point operations) per time step.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0   |
    /// 1/3 | 1/3
    /// 2/3 | -1/3 1
    /// 1   | 1   -1   1
    /// ---------------------
    ///   | 1/8 3/8 3/8 1/8
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples)
    /// 
    name: ThreeEights,
    a: [[0.0, 0.0, 0.0, 0.0],
        [1.0/3.0, 0.0, 0.0, 0.0],
        [-1.0/3.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0, 0.0]],
    b: [1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0],
    c: [0.0, 1.0/3.0, 2.0/3.0, 1.0],
    order: 4,
    stages: 4
);