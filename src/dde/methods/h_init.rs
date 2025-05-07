use crate::{
    dde::DDE,
    alias::NumEvals,
    traits::{CallBackData, Real, State},
};
    
pub fn h_init<const L: usize, T, V, D, F> (
    dde: &F,
    t0_init: T,
    tf_init: T,
    y0_init: &V,
    order: usize, // Method order, 3 for BS23
    rtol_init: T,
    atol_init: T,
    h_min_init: T,
    h_max_init: T,
    phi_init: &impl Fn(T) -> V,
    f0_init: &V,
) -> (T, NumEvals)
where 
    T: Real,
    V: State<T>,
    D: CallBackData,
    F: DDE<L, T, V, D>,
{
    let mut evals = 0;
    let posneg_init = (tf_init - t0_init).signum();
    let n_dim = y0_init.len();

    let mut dnf = T::zero();
    let mut dny = T::zero();
    for n in 0..n_dim {
        let sk = atol_init + rtol_init * y0_init.get(n).abs();
        if sk <= T::zero() { return (h_min_init.abs().max(T::from_f64(1e-6).unwrap()) * posneg_init, evals); }
        dnf += (f0_init.get(n) / sk).powi(2);
        dny += (y0_init.get(n) / sk).powi(2);
    }
    if n_dim > 0 {
        dnf = (dnf / T::from_usize(n_dim).unwrap()).sqrt();
        dny = (dny / T::from_usize(n_dim).unwrap()).sqrt();
    } else { // Scalar case
            dnf = dnf.sqrt();
            dny = dny.sqrt();
    }


    let mut h = if dnf <= T::from_f64(1.0e-10).unwrap() || dny <= T::from_f64(1.0e-10).unwrap() {
        T::from_f64(1.0e-6).unwrap()
    } else {
        (dny / dnf) * T::from_f64(0.01).unwrap()
    };
    h = h.min(h_max_init.abs());
    h *= posneg_init;

    let mut y1 = *y0_init + *f0_init * h;
    let mut t1 = t0_init + h;
    let mut f1 = V::zeros();
    
    let mut current_lags_init = [T::zero(); L];
    let mut yd_init = [V::zeros(); L];

    // Ensure initial step's delayed points are valid
    if L > 0 {
        loop { // Adjust h if t1 - lag is "beyond" t0 for phi
            dde.lags(t1, &y1, &mut current_lags_init);
            let mut reduce_h_for_lag = false;
            let mut h_candidate_from_lag = h.abs();

            for i in 0..L {
                if current_lags_init[i] <= T::zero() { /* error or skip */ continue; }
                let t_delayed = t1 - current_lags_init[i];
                if (t_delayed - t0_init) * posneg_init < -T::default_epsilon() { // t_delayed is "before" t0
                    // This is fine, phi will be used.
                } else { // t_delayed is "at or after" t0. This means current h is too large.
                    // We need t1 - lag <= t0  => h + t0 - lag <= t0 => h <= lag
                    h_candidate_from_lag = h_candidate_from_lag.min(current_lags_init[i].abs() * T::from_f64(0.99).unwrap() ); // Reduce h to be less than this lag
                    reduce_h_for_lag = true;
                }
            }

            if reduce_h_for_lag && h_candidate_from_lag < h.abs() {
                h = h_candidate_from_lag * posneg_init;
                if h.abs() < h_min_init.abs() { h = h_min_init * posneg_init; } // Respect h_min
                if h.abs() < T::default_epsilon() { // Avoid zero step
                    return (h_min_init.abs().max(T::from_f64(1e-6).unwrap()) * posneg_init, evals);
                }
                y1 = *y0_init + *f0_init * h;
                t1 = t0_init + h;
                // Loop again with new h
            } else {
                break; // h is fine regarding lags for phi
            }
        }
        // Populate yd_init for the diff call
        dde.lags(t1, &y1, &mut current_lags_init); // Recalculate lags with final t1, y1
        for i in 0..L {
                let t_delayed = t1 - current_lags_init[i];
                yd_init[i] = phi_init(t_delayed);
        }
    }
    
    dde.diff(t1, &y1, &yd_init, &mut f1);
    evals += 1;

    let mut der2 = T::zero();
    for n in 0..n_dim {
        let sk = atol_init + rtol_init * y0_init.get(n).abs();
        if sk <= T::zero() { der2 = T::infinity(); break; }
        der2 += ((f1.get(n) - f0_init.get(n)) / sk).powi(2);
    }
    if n_dim > 0 {
        der2 = (der2 / T::from_usize(n_dim).unwrap()).sqrt() / h.abs();
    } else { // Scalar
        der2 = der2.sqrt() / h.abs();
    }


    let der12 = dnf.max(der2);
    let h1 = if der12 <= T::from_f64(1.0e-15).unwrap() {
        h.abs().max(T::from_f64(1.0e-6).unwrap()) * T::from_f64(0.1).unwrap()
    } else {
        let order_t = T::from_usize(order + 1).unwrap(); // order is method order (3 for BS23)
        (T::from_f64(0.01).unwrap() / der12).powf(T::one() / order_t)
    };

    h = h.abs().min(h1);
    h = h.min(h_max_init.abs());
    if h_min_init.abs() > T::zero() {
        h = h.max(h_min_init.abs());
    }
    (h * posneg_init, evals)
}