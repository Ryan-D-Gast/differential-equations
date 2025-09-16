//! Solout implementation which takes a Event object and uses it to detect events

use super::*;

pub struct EventConfig {
    /// Direction of zero crossing to detect
    pub direction: CrossingDirection,
    /// Number of events before termination
    pub terminate: Option<u32>,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            direction: CrossingDirection::Both,
            terminate: None,
        }
    }
}

impl EventConfig {
    /// Create a new EventConfig with specified direction and termination count
    pub fn new(direction: impl Into<CrossingDirection>, terminate: Option<u32>) -> Self {
        Self {
            direction: direction.into(),
            terminate,
        }
    }

    pub fn direction(mut self, direction: impl Into<CrossingDirection>) -> Self {
        self.direction = direction.into();
        self
    }

    /// Set the number of events before termination
    pub fn terminate_after(mut self, count: u32) -> Self {
        self.terminate = Some(count);
        self
    }

    /// Set to terminate after the first event
    pub fn terminal(mut self) -> Self {
        self.terminate = Some(1);
        self
    }
}

pub trait Event<T: Real = f64, Y: State<T> = f64> {
    /// Configure the event detection parameters (called once at initialization).
    fn config(&self) -> EventConfig {
        EventConfig::default()
    }

    /// Event function g(t,y) whose zero crossings are detected.
    fn event(&self, t: T, y: &Y) -> T;
}

/// Solout implementation that evaluates user-provided Event objects similar to SciPy events.
///
/// The `EventSolout` monitors the sign of `event.event(t, y)` across solver steps. When a sign
/// change consistent with the configured `CrossingDirection` is detected, a Brent-Dekker root
/// finding procedure is applied (using the solver's interpolation) to locate the event time with
/// high accuracy. The event point `(t_event, y_event)` is then appended to the solution. Depending
/// on the `EventConfig::terminate` setting the integration may terminate after collecting the
/// specified number of events.
pub struct EventSolout<'a, T: Real, Y: State<T>, E: Event<T, Y>> {
    /// User provided event object implementing `Event`
    event: &'a E,
    /// Configuration (direction filtering and termination count)
    config: EventConfig,
    /// Last event function value g(t_prev, y_prev)
    last_g: Option<T>,
    /// Number of events detected so far
    event_count: u32,
    /// Integration direction cached (+1 or -1)
    direction: T,
    /// Tolerance factor for root finding termination
    rel_tol: T,
    /// Absolute tolerance floor for root search
    abs_tol: T,
    /// State type marker
    _marker: std::marker::PhantomData<Y>,
}

impl<'a, T: Real, Y: State<T>, E: Event<T, Y>> EventSolout<'a, T, Y, E> {
    pub fn new(event: &'a E, t0: T, tf: T) -> Self {
        let direction = (tf - t0).signum();
        let config = event.config();
        EventSolout {
            event,
            config,
            last_g: None,
            event_count: 0,
            direction,
            rel_tol: T::from_f64(1e-12).unwrap_or(T::default_epsilon()),
            abs_tol: T::from_f64(1e-14).unwrap_or(T::default_epsilon()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Brent-Dekker root finding for locating g(t)=0 within [a,b] where g(a)*g(b) <= 0.
    /// Uses interpolator to obtain y(t) for evaluating g(t) = event.event(t, y(t)).
    fn brent_dekker<I>(
        &mut self,
        mut a: T,
        mut b: T,
        mut fa: T,
        mut fb: T,
        interpolator: &mut I,
    ) -> Option<T>
    where
        I: Interpolation<T, Y>,
    {
        // Ensure that |f(a)| < |f(b)| swapping if necessary
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut e = d;

        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        let half = one / two;
        let three = T::from_f64(3.0).unwrap();

        let max_iter = 50u32;
        for _ in 0..max_iter {
            if fb == T::zero() {
                return Some(b);
            }
            if fa.signum() == fb.signum() {
                // Rename a -> c
                a = c;
                fa = fc;
                c = b;
                fc = fb;
                d = b - a;
                e = d;
            }
            if fa.abs() < fb.abs() {
                c = b;
                b = a;
                a = c;
                fc = fb;
                fb = fa;
                fa = fc;
            }

            // Convergence check
            let tol = self.abs_tol.max(self.rel_tol * b.abs());
            let m = half * (a - b);
            if m.abs() <= tol || fb == T::zero() {
                return Some(b);
            }

            // Attempt inverse quadratic interpolation or secant
            let mut use_bisection = true;
            if e.abs() > tol && fa.abs() > fb.abs() {
                // Inverse quadratic interpolation
                let s = fb / fa;
                let p;
                let q;
                if a == c {
                    // Secant method
                    p = two * m * s;
                    q = one - s;
                } else {
                    // Inverse quadratic interpolation
                    let q1 = fa / fc;
                    let r = fb / fc;
                    p = s * (two * m * q1 * (q1 - r) - (b - a) * (r - one));
                    q = (q1 - one) * (r - one) * (s - one);
                }
                let mut q_mod = q;
                let mut p_mod = p;
                if q_mod > T::zero() {
                    p_mod = -p_mod;
                } else {
                    q_mod = -q_mod;
                }
                // Accept interpolation only if conditions satisfied
                if (two * p_mod).abs() < (three * m * q_mod - (tol * q_mod).abs())
                    && p_mod < (e * half * q_mod).abs()
                {
                    e = d;
                    d = p_mod / q_mod;
                    use_bisection = false;
                }
            }
            if use_bisection {
                d = m;
                e = m;
            }
            // Move last best b
            a = b;
            fa = fb;
            if d.abs() > tol {
                b = b + d;
            } else {
                b = b + if m > T::zero() { tol } else { -tol };
            }
            // Evaluate at new b via interpolation
            let yb = interpolator.interpolate(b).ok()?;
            fb = self.event.event(b, &yb);
            c = a;
            fc = fa;
        }
        None
    }
}

impl<'a, T, Y, E> Solout<T, Y> for EventSolout<'a, T, Y, E>
where
    T: Real,
    Y: State<T>,
    E: Event<T, Y>,
{
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Y,
        y_prev: &Y,
        interpolator: &mut I,
        solution: &mut Solution<T, Y>,
    ) -> ControlFlag<T, Y>
    where
        I: Interpolation<T, Y>,
    {
        // Evaluate event function at current endpoint
        let g_curr = self.event.event(t_curr, y_curr);

        // Initialize previous value if first call
        let g_prev = match self.last_g {
            Some(g) => g,
            None => {
                let g0 = self.event.event(t_prev, y_prev);
                self.last_g = Some(g0);
                // We don't attempt detection on first initialization
                self.last_g = Some(g_curr);
                return ControlFlag::Continue;
            }
        };

        // Detect sign change according to direction config
        let zero = T::zero();
        let sign_change = g_prev.signum() != g_curr.signum();

        let direction_ok = match self.config.direction {
            CrossingDirection::Both => sign_change,
            CrossingDirection::Positive => sign_change && g_prev < zero && g_curr >= zero,
            CrossingDirection::Negative => sign_change && g_prev > zero && g_curr <= zero,
        };

        if direction_ok {
            // Root find for precise event time
            let (mut a, mut b, mut fa, mut fb) = (t_prev, t_curr, g_prev, g_curr);
            // Ensure bracket ordering consistent with integration direction
            if (self.direction > zero && a > b) || (self.direction < zero && a < b) {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }

            // Only proceed if fa*fb <= 0
            if fa * fb <= zero {
                if let Some(t_event) = self.brent_dekker(a, b, fa, fb, interpolator) {
                    let y_event = interpolator.interpolate(t_event).unwrap();
                    // Avoid duplicate near-equal times
                    let push_point = match solution.t.last() {
                        Some(&last_t) => (t_event - last_t).abs() > self.abs_tol,
                        None => true,
                    };
                    if push_point {
                        solution.push(t_event, y_event);
                    }
                    self.event_count += 1;

                    if let Some(limit) = self.config.terminate {
                        if self.event_count >= limit {
                            self.last_g = Some(g_curr);
                            return ControlFlag::Terminate;
                        }
                    }
                }
            }
        }

        self.last_g = Some(g_curr);
        ControlFlag::Continue
    }
}

/// Wrapper solout that decorates an existing solout with event detection.
pub struct EventWrappedSolout<'a, T: Real, Y: State<T>, O, E>
where
    O: Solout<T, Y>,
    E: Event<T, Y>,
{
    base: O,
    event: &'a E,
    config: EventConfig,
    last_g: Option<T>,
    event_count: u32,
    direction: T,
    rel_tol: T,
    abs_tol: T,
    _marker: std::marker::PhantomData<Y>,
}

impl<'a, T: Real, Y: State<T>, O, E> EventWrappedSolout<'a, T, Y, O, E>
where
    O: Solout<T, Y>,
    E: Event<T, Y>,
{
    pub fn new(base: O, event: &'a E, t0: T, tf: T) -> Self {
        let config = event.config();
        EventWrappedSolout {
            base,
            event,
            config,
            last_g: None,
            event_count: 0,
            direction: (tf - t0).signum(),
            rel_tol: T::from_f64(1e-12).unwrap_or(T::default_epsilon()),
            abs_tol: T::from_f64(1e-14).unwrap_or(T::default_epsilon()),
            _marker: std::marker::PhantomData,
        }
    }

    fn detect_event<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Y,
        y_prev: &Y,
        interpolator: &mut I,
        solution: &mut Solution<T, Y>,
    ) -> ControlFlag<T, Y>
    where
        I: Interpolation<T, Y>,
    {
        let g_curr = self.event.event(t_curr, y_curr);
        let g_prev = match self.last_g {
            Some(g) => g,
            None => {
                let g0 = self.event.event(t_prev, y_prev);
                self.last_g = Some(g0);
                self.last_g = Some(g_curr);
                return ControlFlag::Continue;
            }
        };

        let zero = T::zero();
        let sign_change = g_prev.signum() != g_curr.signum();
        let direction_ok = match self.config.direction {
            CrossingDirection::Both => sign_change,
            CrossingDirection::Positive => sign_change && g_prev < zero && g_curr >= zero,
            CrossingDirection::Negative => sign_change && g_prev > zero && g_curr <= zero,
        };
        if direction_ok {
            let (mut a, mut b, mut fa, mut fb) = (t_prev, t_curr, g_prev, g_curr);
            if (self.direction > zero && a > b) || (self.direction < zero && a < b) {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
            if fa * fb <= zero {
                if let Some(t_event) = self.brent_dekker(a, b, fa, fb, interpolator) {
                    let y_event = interpolator.interpolate(t_event).unwrap();
                    let push_point = match solution.t.last() {
                        Some(&last_t) => (t_event - last_t).abs() > self.abs_tol,
                        None => true,
                    };
                    if push_point {
                        solution.push(t_event, y_event);
                    }
                    self.event_count += 1;
                    if let Some(limit) = self.config.terminate {
                        if self.event_count >= limit {
                            self.last_g = Some(g_curr);
                            return ControlFlag::Terminate;
                        }
                    }
                }
            }
        }
        self.last_g = Some(g_curr);
        ControlFlag::Continue
    }

    fn brent_dekker<I>(
        &mut self,
        mut a: T,
        mut b: T,
        mut fa: T,
        mut fb: T,
        interpolator: &mut I,
    ) -> Option<T>
    where
        I: Interpolation<T, Y>,
    {
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut e = d;
        let one = T::one();
        let two = T::from_f64(2.0).unwrap();
        let half = one / two;
        let three = T::from_f64(3.0).unwrap();
        for _ in 0..50u32 {
            if fb == T::zero() {
                return Some(b);
            }
            if fa.signum() == fb.signum() {
                a = c;
                fa = fc;
                c = b;
                fc = fb;
                d = b - a;
                e = d;
            }
            if fa.abs() < fb.abs() {
                c = b;
                b = a;
                a = c;
                fc = fb;
                fb = fa;
                fa = fc;
            }
            let tol = self.abs_tol.max(self.rel_tol * b.abs());
            let m = half * (a - b);
            if m.abs() <= tol || fb == T::zero() {
                return Some(b);
            }
            let mut use_bis = true;
            if e.abs() > tol && fa.abs() > fb.abs() {
                let s = fb / fa;
                let p;
                let q;
                if a == c {
                    p = two * m * s;
                    q = one - s;
                } else {
                    let q1 = fa / fc;
                    let r = fb / fc;
                    p = s * (two * m * q1 * (q1 - r) - (b - a) * (r - one));
                    q = (q1 - one) * (r - one) * (s - one);
                }
                let mut q_mod = q;
                let mut p_mod = p;
                if q_mod > T::zero() {
                    p_mod = -p_mod;
                } else {
                    q_mod = -q_mod;
                }
                if (two * p_mod).abs() < (three * m * q_mod - (tol * q_mod).abs())
                    && p_mod < (e * half * q_mod).abs()
                {
                    e = d;
                    d = p_mod / q_mod;
                    use_bis = false;
                }
            }
            if use_bis {
                d = m;
                e = m;
            }
            a = b;
            fa = fb;
            b = if d.abs() > tol {
                b + d
            } else {
                b + if m > T::zero() { tol } else { -tol }
            };
            let yb = interpolator.interpolate(b).ok()?;
            fb = self.event.event(b, &yb);
            c = a;
            fc = fa;
        }
        None
    }
}

impl<'a, T, Y, O, E> Solout<T, Y> for EventWrappedSolout<'a, T, Y, O, E>
where
    T: Real,
    Y: State<T>,
    O: Solout<T, Y>,
    E: Event<T, Y>,
{
    fn solout<I>(
        &mut self,
        t_curr: T,
        t_prev: T,
        y_curr: &Y,
        y_prev: &Y,
        interpolator: &mut I,
        solution: &mut Solution<T, Y>,
    ) -> ControlFlag<T, Y>
    where
        I: Interpolation<T, Y>,
    {
        let flag = self
            .base
            .solout(t_curr, t_prev, y_curr, y_prev, interpolator, solution);
        if let ControlFlag::Terminate = flag {
            return flag;
        }
        let evt_flag = self.detect_event(t_curr, t_prev, y_curr, y_prev, interpolator, solution);
        match evt_flag {
            ControlFlag::Terminate => ControlFlag::Terminate,
            _ => flag,
        }
    }
}
