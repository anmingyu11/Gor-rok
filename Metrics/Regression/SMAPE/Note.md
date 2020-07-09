# Symmetric mean absolute percentage error

**Symmetric mean absolute percentage error (SMAPE** or **sMAPE)** is an accuracy measure based on percentage (or relative) errors. It is usually defined as follows:
$$
SMAPE = \frac{100%}{n}\sum^{n}_{t=1}\frac{|F_t - A_t|}{(A_t| + |F_t|)/2}
$$
where $A_t$ is the actual value and $F_t$ is the forecast value.



The absolute difference between $A_t$ and $F_t$ is divided by half the sum of absolute values of the actual value $A_t$ and the forecast value $F_t$. The value of this calculation is summed for every fitted point *t* and divided again by the number of fitted points *n*.

The earliest reference to similar formula appears to be Armstrong (1985, p. 348) where it is called "adjusted MAPE" and is defined without the absolute values in denominator. It has been later discussed, modified and re-proposed by Flores (1986).

Armstrong's original definition is as follows:
$$
SMAPE = \frac{1}{n}\sum^{n}_{t=1}\frac{|F_t - A_t|}{(|F_t|+|A_t|)/2}
$$
The problem is that it can be negative (if ${A_{t}+F_{t}<0}$ or even undefined (if ${A_{t}+F_{t}=0}$). Therefore the currently accepted version of SMAPE assumes the absolute values in the denominator.

In contrast to the mean absolute percentage error, SMAPE has both a lower bound and an upper bound. Indeed, the formula above provides a result between 0% and 200%. However a percentage error between 0% and 100% is much easier to interpret. That is the reason why the formula below is often used in practice (i.e. no factor 0.5 in denominator):
$$
SMAPE = \frac{100%}{n}\sum_{t=1}^{n}\frac{|F_t - A_t|}{|A_t|+|F_t|}
$$
One supposed problem with **SMAPE** is that it is not symmetric since over- and under-forecasts are not treated equally. This is illustrated by the following example by applying the second **SMAPE** formula:

- Over-forecasting: $A_t$ = 100 and $F_t$ = 110 give SMAPE = 4.76%
- Under-forecasting: $A_t$ = 100 and $F_t$ = 90 give SMAPE = 5.26%.

However, one should only expect this type of symmetry for measures which are entirely difference-based and not relative (such as mean squared error and mean absolute deviation).

There is a third version of SMAPE, which allows to measure the direction of the bias in the data by generating a positive and a negative error on line item level. Furthermore it is better protected against outliers and the bias effect mentioned in the previous paragraph than the two other formulas. The formula is:
$$
SMAPE = \frac{\sum_{t=1}^{n}|F_t - A_t|}{\sum_{t=1}^{n}(A_t + F_t)}
$$

A limitation to SMAPE is that if the actual value or forecast value is 0, the value of error will boom up to the upper-limit of error. (200% for the first formula and 100% for the second formula).

Provided the data are strictly positive, a better measure of relative accuracy can be obtained based on the log of the accuracy ratio: $log(F_t / A_t)$ This measure is easier to analyse statistically, and has valuable symmetry and unbiasedness properties. When used in constructing forecasting models the resulting prediction corresponds to the geometric mean (Tofallis, 2015).