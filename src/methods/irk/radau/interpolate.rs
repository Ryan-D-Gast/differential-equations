//! Interpolation for Radau5

use crate::{
    error::Error,
    interpolate::Interpolation,
    methods::irk::radau::Radau5,
    traits::{Real, State},
};

impl<E, T: Real, Y: State<T>> Interpolation<T, Y> for Radau5<E, T, Y> {
    /// Dense output on [t_prev, t].
    fn interpolate(&mut self, t_interp: T) -> Result<Y, Error<T, Y>> {
        if t_interp < self.t_prev || t_interp > self.t {
            return Err(Error::OutOfBounds {
                t_interp,
                t_prev: self.t_prev,
                t_curr: self.t,
            });
        }

        // Collocation polynomial parameter s = (t - t_curr) / h_prev, so s in [-1, 0]
        let s = (t_interp - self.t) / self.h_prev;

        // y = CONT0 + S*(C1 + (S-C2M1)*(C2 + (S-C1M1)*C3))
        let term3 = self.cont[3].scaled(s - self.c1m1);
        let term2 = self.cont[2].plus_scaled(T::one(), &term3).scaled(s - self.c2m1);
        let term1 = self.cont[1].plus_scaled(T::one(), &term2).scaled(s);
        let y = self.cont[0].plus_scaled(T::one(), &term1);

        Ok(y)
    }
}
