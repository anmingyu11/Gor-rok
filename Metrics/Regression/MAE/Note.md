# Mean absolute error

在统计中，平均绝对误差（MAE）是表示同一现象的成对观测值之间误差的度量。  $Y$ 与 $X$ 的示例包括比较预测值与观察值，比较 subsequent time 与 initial time，以及一种测量技术与另一种测量技术。 

MAE的计算公式为：

因此，它是绝对误差${| e_ {i} | = | y_ {i} -x_ {i} |}$的算术平均值，其中${y_ {i}}$是预测，${x_ {i}}$是​真实值。注意，替代公式可以包括相对频率作为权重因子。MAE 使用与被测数据相同的量纲。这称为量纲相关的精度度量，因此不能用于使用不同量纲 进行Series-Series之间的比较。平均绝对误差是时间序列分析中预测误差的常用度量，有时与平均绝对偏差的更标准定义混淆使用。同样的困惑更普遍地存在。

## Quantity disagreement and allocation disagreement

It is possible to express MAE as the sum of two components: Quantity Disagreement and Allocation Disagreement. Quantity Disagreement is the absolute value of the Mean Error given by:

#### Quantity Disagreement

$$
ME = \frac{\sum_{i=1}^{n}y_i-x_i}{n}
$$

It is also possible to identify the types of difference by looking at an ${(x,y)}$ plot. Quantity difference exists when the average of the X values does not equal the average of the Y values. 

#### Allocation Disagreement

Allocation Disagreement is MAE minus Quantity Disagreement.

Allocation difference exists if and only if points reside on both sides of the identity line.

> 当且仅当点位于标识线的两侧时才存在分配差异。

![](/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MAE/1.png)

MAE is not identical to RMSE (root-mean square error), but some researchers report and interpret RMSE as if RMSE reflects the measurement that MAE gives. 

MAE is conceptually simpler and more interpretable than RMSE. 

MAE does not require the use of squares or square roots.

The use of squared distances hinders the interpretation of RMSE. 

MAE is simply the average absolute vertical or horizontal distance between each point in a scatter plot and the $Y=X$ line. 

In other words, MAE is the average absolute difference between $X$ and$Y$.

**MAE is fundamentally easier to understand than the square root of the average of the sum of squared deviations. **

Furthermore, each error contributes to MAE in proportion to the absolute value of the error, which is not true for RMSE. 

## Optimality property

The mean absolute error of a real variable $c$ with respect to the random variable X is${E(\left|X-c\right|)\}}$ , Provided that the probability distribution of X is such that the above expectation exists, then m is a median of X if and only if m is a minimizer of the mean absolute error with respect to X.

In particular, m is a sample median if and only if m minimizes the arithmetic mean of the absolute deviations.

More generally, a median is defined as a minimum of

$$
E(|X-c|-|X|)
$$
as discussed at Multivariate median (and specifically at Spatial median).

This optimization-based definition of the median is useful in statistical data-analysis, for example, in k-medians clustering.

> 最优属性
>
> 实变量c相对于随机变量X的平均绝对误差是
> $$
> {E(\left|X-c\right|)\}}
> $$
> 假设 $X$ 的概率分布是上述期望存在的，那么$m$是$X$的中值当且仅当$m$是关于$X$的绝对平均误差的最小值。
>
> 特别是，$m$是样本中值当且仅当$m$使绝对值的算术平均值最小时。
>
> 更一般地，中值定义为的最小值
>
> 如多元中位数(特别是空间中位数)讨论的那样。
>
> 这种基于优化的中值定义在统计数据分析中很有用，例如在$k$中值聚类中。

