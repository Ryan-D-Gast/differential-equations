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
        let mut y = self.y_prev.zeros_like();

        let dim = self.y_prev.len();
        let mut cont0 = vec![T::zero(); dim];
        let mut cont1 = vec![T::zero(); dim];
        let mut cont2 = vec![T::zero(); dim];
        let mut cont3 = vec![T::zero(); dim];
        let mut values = vec![T::zero(); dim];
        self.cont[0].write_to_slice(&mut cont0);
        self.cont[1].write_to_slice(&mut cont1);
        self.cont[2].write_to_slice(&mut cont2);
        self.cont[3].write_to_slice(&mut cont3);
        for i in 0..dim {
            // CONT0 + S*(C1 + (S-C2M1)*(C2 + (S-C1M1)*C3))
            values[i] = cont0[i]
                + s * (cont1[i] + (s - self.c2m1) * (cont2[i] + (s - self.c1m1) * cont3[i]));
        }
        y.read_from_slice(&values);

        Ok(y)
    }
}
