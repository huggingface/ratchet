use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, Slice};
use num_traits::{Float, FromPrimitive};

pub trait NDArrayExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn logsumexp(&self, axis: usize) -> A
    where
        A: Float + FromPrimitive,
        D: RemoveAxis;

    fn log_softmax(&self, axis: usize) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis;

    fn softmax(&self, axis: usize) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis;
    fn pad(&self, pad_width: Vec<[usize; 2]>, const_value: A) -> Array<A, D>;
}

impl<A, S, D> NDArrayExt<A, S, D> for ArrayBase<S, D>
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn logsumexp(&self, axis: usize) -> A
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
    {
        self.lanes(Axis(axis))
            .into_iter()
            .fold(A::neg_infinity(), |log_sum_exp, lane| {
                let max = lane.fold(A::neg_infinity(), |a, &b| a.max(b));
                let log_sum_exp_lane = max
                    + lane
                        .mapv(|x| {
                            if x.is_infinite() {
                                A::zero()
                            } else {
                                (x - max).exp()
                            }
                        })
                        .sum()
                        .ln();
                A::max(log_sum_exp, log_sum_exp_lane)
            })
    }

    fn log_softmax(&self, axis: usize) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
    {
        let mut output = self.to_owned();
        output
            .lanes_mut(Axis(axis))
            .into_iter()
            .for_each(|mut lane| {
                let max = lane.fold(A::neg_infinity(), |a, &b| a.max(b));
                lane.mapv_inplace(|x| x - max);
                let log_sum_exp = lane
                    .mapv(|x| if x.is_infinite() { A::zero() } else { x.exp() })
                    .sum()
                    .ln();
                lane.mapv_inplace(move |x| x - log_sum_exp);
            });
        output
    }

    fn softmax(&self, axis: usize) -> Array<A, D>
    where
        A: Float + FromPrimitive,
        D: RemoveAxis,
    {
        let mut output = self.to_owned();
        output
            .lanes_mut(Axis(axis))
            .into_iter()
            .for_each(|mut lane| {
                let max = lane.fold(A::neg_infinity(), |a, &b| a.max(b));
                lane.mapv_inplace(|x| x - max);
                let sum_exp = lane.mapv(|x| x.exp()).sum();
                lane.mapv_inplace(move |x| x.exp() / sum_exp);
            });
        output
    }

    fn pad(&self, pad_width: Vec<[usize; 2]>, const_value: A) -> Array<A, D> {
        assert_eq!(
            self.ndim(),
            pad_width.len(),
            "Array ndim must match length of `pad_width`."
        );

        // Compute shape of final padded array.
        let mut padded_shape = self.raw_dim();
        for (ax, (&ax_len, &[pad_lo, pad_hi])) in self.shape().iter().zip(&pad_width).enumerate() {
            padded_shape[ax] = ax_len + pad_lo + pad_hi;
        }

        let mut padded = Array::from_elem(padded_shape, const_value);
        let padded_dim = padded.raw_dim();
        {
            // Select portion of padded array that needs to be copied from the
            // original array.
            let mut orig_portion = padded.view_mut();
            for (ax, &[pad_lo, pad_hi]) in pad_width.iter().enumerate() {
                orig_portion.slice_axis_inplace(
                    Axis(ax),
                    Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
                );
            }
            // Copy the data from the original array.
            orig_portion.assign(self);
        }
        padded
    }
}
