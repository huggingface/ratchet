use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis};
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
}

impl<A, S, D> NDArrayExt<A, S, D> for ArrayBase<S, D>
where
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
}
