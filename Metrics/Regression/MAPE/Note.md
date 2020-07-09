# Mean absolute percentage error

The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of prediction accuracy of a forecasting method in statistics, for example in trend estimation, also used as a loss function for regression problems in machine learning. It usually expresses the accuracy as a ratio defined by the formula:
$$
M= \frac{1}{n}\sum_{t=1}^{n}|\frac{A_t - F_t}{A_t}|
$$
where $A_t$ is the actual value and $F_t$ is the forecast value.

The MAPE is also sometimes reported as a percentage, which is the above equation multiplied by 100. 

The difference between $A_t$ and $F_t$ is divided by the actual value $A_t$ again.

The absolute value in this calculation is summed for every forecasted point in time and divided by the number of fitted points *n*. Multiplying by 100% makes it a percentage error.

## MAPE in regression problems

Mean absolute percentage error is commonly used as a loss function for regression problems and in model evaluation, because of its very intuitive interpretation in terms of relative error.

### Definition

Consider a standard regression setting in which the data are fully described by a random pair ${Z=(X,Y)}$ with values in ${ \mathbb {R} ^{d}\times \mathbb {R} }$, and *n* i.i.d. copies ${(X_{1},Y_{1}),...,(X_{n},Y_{n})}$ of ${(X,Y)}$.

Regression models aims at finding a good model for the pair, that is a [measurable function](https://en.wikipedia.org/wiki/Measurable_function) $g$ from ${ \mathbb {R} ^{d}}$ to ${ \mathbb {R} }$ such that ${g(X)}$ is close to $Y$.

In the classical regression setting , the closeness of ${g(X)}$ to $Y$ is measured via the $L2$ risk, also called the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE). 

In the MAPE regression context , the closeness of ${g(X)}$ to $Y$ is measured via the MAPE , and the aim of MAPE regressions is to find a model ${g_{MAPE}}$ such that:
$$
{g_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\mathbb {E} \left[\left|{\frac {g(X)-Y}{Y}}\right||X=x\right]}
$$
where ${{\mathcal {G}}}$ is the class of models considered (e.g. linear models).

**In practice**

In practice ${g_{MAPE}(x)}$ can be estimated by the [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) strategy, leading to
$$
{{\widehat {g}}_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\sum _{i=1}^{n}\left|{\frac {g(X_{i})-Y_{i}}{Y_{i}}}\right|}
$$
From a practical point of view, the use of the MAPE as a quality function for regression model is equivalent to doing weighted [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE) regression, also known as [quantile regression](https://en.wikipedia.org/wiki/Quantile_regression). This property is trivial since
$$
{{\widehat {g}}_{MAPE}(x)=\arg \min _{g\in {\mathcal {G}}}\sum _{i=1}^{n}\omega (Y_{i})\left|g(X_{i})-Y_{i}\right|{\mbox{ with }}\omega (Y_{i})=\left|{\frac {1}{Y_{i}}}\right|}
$$
As a consequence, the use of the MAPE is very easy in practice, for example using existing libraries for quantile regression allowing weights.

### Consistency

The use of the MAPE as a loss function for regression analysis is feasible both on a practical point of view and on a theoretical one, since the existence of an optimal model and the [consistency](https://en.wikipedia.org/wiki/Consistency_(statistics)) of the empirical risk minimization can be proved.

> 无论是从 实践角度 还是从 理论角度 ，将 MAPE 用作损失函数进行回归分析都是可行的，因为可以证明存在最优模型和经验风险最小化的一致性。

## Alternative MAPE definitions

Problems can occur when calculating the MAPE value with a series of small denominators. 

A singularity problem of the form 'one divided by zero' and/or the creation of very large changes in the Absolute Percentage Error , caused by a small deviation in error, can occur.

As an alternative, each actual value ($A_t$) of the series in the original formula can be replaced by the average of all actual values ($\bar A_t$) of that series. 

> 或者，可以将原始公式中该系列的每个实际值（$ A_t $）替换为该系列的所有实际值（$ \bar A_t $）的平均值。

This alternative is still being used for measuring the performance of models that forecast spot electricity prices.

Note that this is equivalent to dividing the sum of absolute differences by the sum of actual values, and is sometimes referred to as WAPE (weighted absolute percentage error).

> 注意，这相当于用绝对值差和除以实际值和，有时称为WAPE(加权绝对百分比误差)。

## Issues

Although the concept of MAPE sounds very simple and convincing, it has major drawbacks(缺陷) in practical application, and there are many studies on shortcomings and misleading results from MAPE.

- It cannot be used if there are zero values (which sometimes happens for example in demand data) because there would be a division by zero.

- For forecasts which are too low the percentage error cannot exceed 100%, but for forecasts which are too high there is no upper limit to the percentage error.

- > 对于太低的预测，误差不能超过100%，但对于太高的预测，误差没有上限。
  >
  > 这里的描述有些问题，应该说对于误差太大的预测，MAPE没有上限

MAPE puts a heavier penalty on negative errors, ${A_{t}<F_{t}}$than on positive errors.

> MAPE对负误差$ {A_ {t} <F_ {t}} $的惩罚要比正误差大。

As a consequence, when MAPE is used to compare the accuracy of prediction methods it is biased in that it will systematically select a method whose forecasts are too low.

> 结果，当使用MAPE来比较预测方法的准确性时，它会产生偏差，因为它将系统地选择预测值太低的方法。
>

This little-known but serious issue can be overcome by using an accuracy measure based on the logarithm of the accuracy ratio (the ratio of the predicted to actual value), given by 
$$
{ \log \left({\frac {\text{predicted}}{\text{actual}}}\right)}
$$

> 这个鲜为人知但严重的问题可以通过使用基于准确度比率(预测值与实际值之比)的对数的准确度度量来克服
> $$
> { \log \left({\frac {\text{predicted}}{\text{actual}}}\right)}
> $$

This approach leads to superior statistical properties and leads to predictions which can be interpreted in terms of the geometric mean.

> 这种方法带来了卓越的统计特性，并产生了可以根据几何平均值进行解释的预测。