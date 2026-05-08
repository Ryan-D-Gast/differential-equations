use crate::traits::{Real, State};

#[inline]
pub fn zeros_like<T: Real, Y: State<T>>(shape: &Y) -> Y {
    shape.zeros_like()
}

#[inline]
pub fn clone_state<T: Real, Y: State<T>>(state: &Y) -> Y {
    state.clone()
}

#[inline]
pub fn copy_from<T: Real, Y: State<T>>(dst: &mut Y, src: &Y) {
    let len = src.len();
    debug_assert_eq!(dst.len(), len);
    for i in 0..len {
        dst.set(i, src.get(i));
    }
}

#[inline]
pub fn fill<T: Real, Y: State<T>>(dst: &mut Y, value: T) {
    for i in 0..dst.len() {
        dst.set(i, value);
    }
}

#[inline]
pub fn axpy<T: Real, Y: State<T>>(dst: &mut Y, alpha: T, x: &Y) {
    debug_assert_eq!(dst.len(), x.len());
    for i in 0..dst.len() {
        dst.set(i, dst.get(i) + alpha * x.get(i));
    }
}

#[inline]
pub fn assign_scaled<T: Real, Y: State<T>>(dst: &mut Y, x: &Y, alpha: T) {
    debug_assert_eq!(dst.len(), x.len());
    for i in 0..dst.len() {
        dst.set(i, x.get(i) * alpha);
    }
}

#[inline]
pub fn scaled<T: Real, Y: State<T>>(shape: &Y, x: &Y, alpha: T) -> Y {
    debug_assert_eq!(shape.len(), x.len());
    let mut out = shape.zeros_like();
    for i in 0..x.len() {
        out.set(i, x.get(i) * alpha);
    }
    out
}

#[inline]
pub fn add<T: Real, Y: State<T>>(lhs: &Y, rhs: &Y) -> Y {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut out = lhs.zeros_like();
    for i in 0..lhs.len() {
        out.set(i, lhs.get(i) + rhs.get(i));
    }
    out
}

#[inline]
pub fn sub<T: Real, Y: State<T>>(lhs: &Y, rhs: &Y) -> Y {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut out = lhs.zeros_like();
    for i in 0..lhs.len() {
        out.set(i, lhs.get(i) - rhs.get(i));
    }
    out
}

#[inline]
pub fn linear_combination<T: Real, Y: State<T>>(shape: &Y, terms: &[(&Y, T)]) -> Y {
    let mut out = shape.zeros_like();
    for (x, alpha) in terms {
        axpy(&mut out, *alpha, x);
    }
    out
}

#[inline]
pub fn set_linear_combination<T: Real, Y: State<T>>(dst: &mut Y, terms: &[(&Y, T)]) {
    fill(dst, T::zero());
    for (x, alpha) in terms {
        axpy(dst, *alpha, x);
    }
}

#[inline]
pub fn from_base_plus<T: Real, Y: State<T>>(base: &Y, terms: &[(&Y, T)]) -> Y {
    let mut out = base.clone();
    for (x, alpha) in terms {
        axpy(&mut out, *alpha, x);
    }
    out
}

#[inline]
pub fn div_scalar<T: Real, Y: State<T>>(x: &Y, divisor: T) -> Y {
    let mut out = x.zeros_like();
    for i in 0..x.len() {
        out.set(i, x.get(i) / divisor);
    }
    out
}

#[inline]
pub fn norm_squared<T: Real, Y: State<T>>(x: &Y) -> T {
    let mut sum = T::zero();
    for i in 0..x.len() {
        let value = x.get(i);
        sum += value * value;
    }
    sum
}

#[inline]
pub fn diff_norm_squared<T: Real, Y: State<T>>(lhs: &Y, rhs: &Y) -> T {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = T::zero();
    for i in 0..lhs.len() {
        let value = lhs.get(i) - rhs.get(i);
        sum += value * value;
    }
    sum
}
