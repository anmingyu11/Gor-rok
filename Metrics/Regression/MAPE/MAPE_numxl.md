# MAPE - Mean Absolute Percentage Error

Calculates the mean absolute percentage error (Deviation) function for the forecast and the eventual outcomes.

## Syntax

#### MAPE(X,Y,Ret_type)

**X** is the original (eventual outcomes) time series sample data (a one dimensional array of cells (e.g. rows or columns)).

**Y** is the forecast time series data (a one dimensional array of cells (e.g. rows or columns)).

**Ret_type** is a switch to select the return output (1=MAPE (default), 2=Symmetric MAPE (SMAPI)).

| Order |  Description   |
| :---: | :------------: |
|   1   | MAPE (default) |
|   2   |     SMAPE      |

## Remarks

1. MAPE is also referred to as MAPD.

2. The time series is homogeneous or equally spaced.

   > 时间序列是齐次的或等间隔的。

3. For a plain MAPE calculation, in the event that an observation value (i.e. xk) is equal to zero, the MAPE function skips that data point.

   > 对于简单的 MAPE 计算，如果观测值（i.e. $x_k$）等于零，则MAPE函数将跳过该数据点。

4. The mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), measures the accuracy of a method for constructing fitted time series values in statistics.

   > 平均绝对百分比误差（MAPE），也称为平均绝对百分比偏差（MAPD），用于度量统计中构造拟合时间序列值的方法的准确性。

5. The two time series must be identical in size.

6. The mean absolute percentage error (MAPE) is defined as follows:
   $$
   MAPE = \frac{100}{N} * \sum_{i=1}^N |\frac{x_i - \hat{x_i}}{x_i}|
   $$
   Where:

   - ${x_i}$ is the actual observations time series
   - ${x^i}$ is the estimated or forecasted time series
   - $N$ is the number of non-missing data points

7. When calculating the average MAPE for a number of time series, you may encounter a problem: a few of the series that have a very high MAPE might distort a comparison between the average MAPE of a time series fitted with one method compared to the average MAPE when using another method.

   > 在计算多个时间序列的平均 MAPE 时，您可能会遇到一个问题：一些具有很高 MAPE 的 series可能会使采用一种方法的时间序列的平均 MAPE 与平均MAPE的比较产生 distort 当使用其他方法时。

8. In order to avoid this problem, other measures have been defined, for example the SMAPE (symmetrical MAPE), weighted absolute percentage error (WAPE), real aggregated percentage error, and relative measure of accuracy (ROMA).

   > 为了避免这个问题，定义了其他度量，例如SMAPE(对称MAPE)、加权绝对百分比误差(WAPE)、真实综合百分比误差和相对精度度量(ROMA)。

9. The symmetrical mean absolute percentage error (SMAPE) is defined as follows:
   $$
   SMAPE = \frac{200}{N} \times \sum_{i=1}^{N} \frac{|x_i - \hat{x_i}|}{|x_i| + |\hat x_i|}
   $$

10. The SMAPE is easier to work with than MAPE, as it has a lower bound of 0% and an upper bound of 200%.

11. The SMAPE does not treat over-forecast and under-forecast equally.

12. For a SMAPE calculation, in the event the sum of the observation and forecast values (i.e. $x_k+\hat{x_k}$) equals zero, the MAPE function skips that data point.

