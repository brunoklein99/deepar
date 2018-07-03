## DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

### Description

This is an implementation of [1704.04110](https://arxiv.org/pdf/1704.04110.pdf).

### What this implementation does *NOT* contain

Two significant pieces are left out at this time, albeit trivial to implement.

1. The joint embedding learning for item categorization
2. The support for the Gaussian Distribution, suitable in forecasting of real valued timeseries.

* If you decide to implement the Gaussian Distribution, mind the rescaling of the distribution parameters. Refer to the paper.

### Results

Since the paper does not provide quantitative results, we ran the tests with the `carparts` dataset on Amazon's Sagemaker. All the pre-processing & train/valid split was done exactly like stated in the paper.

#### SageMaker's output (single epoch)

```
[07/01/2018 14:22:34 INFO 139862447138624] #test_score (algo-1, wQuantileLoss[0.5]): 1.12679
[07/01/2018 14:22:34 INFO 139862447138624] #test_score (algo-1, mean_wQuantileLoss): 1.13427
[07/01/2018 14:22:34 INFO 139862447138624] #test_score (algo-1, wQuantileLoss[0.9]): 1.14175
[07/01/2018 14:22:34 INFO 139862447138624] #test_score (algo-1, RMSE): 1.07522369541
```

Notice, however, that these metrics are taken computing the ground truths in relation to the average of 50 _samples_

Our implementation is able to achieve RMSE 0.9215
