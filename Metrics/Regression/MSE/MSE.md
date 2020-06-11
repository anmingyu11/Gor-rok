MSE 是风险函数，对应于平方误差损失的期望值。

MSE 几乎总是严格为正数。

MSE是误差$e = (y'-y)$的二阶原点矩，因此结合了estimator 的预测值的方差和与真实值的偏差
$$
MSE(\hat \theta) = Var_{\theta}(\hat \theta) + Bias(\hat \theta,\theta)^2
$$
![](/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MSE/MSE_Var_Bias.svg)

但在实际建模中，MSE可以描述为模型方差、模型偏差和不可约不确定性的增加。

MSE为零，表示估算器$\hat \theta$预测参数的观测值$\theta$完美的精度是理想选择，但通常是不可能的。

MSE的值可用于比较目的。可以使用其MSE来比较**两个或多个统计模型**，以衡量它们对一组给定观察结果的解释程度：在所有无偏估计量中方差最小的无偏估计量（从统计模型估计）是最佳无偏估计量或MVUE （最小方差无偏估计器）。

最小化MSE是选择 estimator 的关键标准：请参见[最小化均方误差](https://en.wikipedia.org/wiki/Minimum_mean-square_error)。在无偏估计量中，最小化MSE等效于使方差最小化，而做到这一点的[估计量](https://en.wikipedia.org/wiki/Minimum_variance_unbiased_estimator)就是[最小方差无偏估计量](https://en.wikipedia.org/wiki/Minimum_variance_unbiased_estimator)。但是，有偏估计器的MSE可能较低；见[估计偏差](https://en.wikipedia.org/wiki/Estimator_bias)。

