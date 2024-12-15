struct Linear<const IN: usize, const OUT: usize> {
    weight: [[f32; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    pub fn new(weight: [[f32; IN]; OUT]) -> Self {
        Self { weight }
    }

    pub fn forward<const B: usize>(&self, xout: &mut [[f32; OUT]; B], xin: &[[f32; IN]; B]) {
        for (xout, xin) in xout.iter_mut().zip(xin) {
            for i in 0..OUT {
                let mut sum = 0.0;
                for j in 0..IN {
                    sum += self.weight[i][j] * xin[j];
                }
                xout[i] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        #[rustfmt::skip]
        let linear = Linear::new([
            [1.0, 2.0],
            [3.0, 4.0]
        ]);
        #[rustfmt::skip]
        let x = [
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let mut y = [[0.0, 0.0], [0.0, 0.0]];
        linear.forward(&mut y, &x);
        assert!(y == [[5.0, 11.0], [11.0, 25.0]]);
    }
}
