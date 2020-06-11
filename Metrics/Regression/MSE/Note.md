均方误差

Mean Square Error

范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；误差越大，该值越大。
$$
\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.
$$

--------------

<img src="/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MSE/MSE_Var_Bias.svg" style="zoom:200%;" />

## Wiki 上的

>https://en.wikipedia.org/wiki/Mean_squared_error

**MSE与RMSE的区别仅在于对量纲是否敏感**