## MAE

$$
MAE = \frac{1}{N}\sum_{i=1}^{N} |y_i'-y_i|
$$

## MSE

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i'-y_i)^2
$$

## RMSE

$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i'-y_i)^2}
$$

## RMSLE

$$
RMSLE = \sqrt{\frac{1}{N}\sum_{i=1}^N(log(y_i + 1) - log(y_i' + 1))^2}
$$

## R-square

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\\= 1 - \frac{\sum_{i=1}^{N}(y_i - y_i')^2}{\sum_{i=1}^{N}(y_i-\bar y)^2}
$$

## MAPE

$$
MAPE = \frac{1}{N}\sum_{i=1}^{N}|\frac{y_i'-y_i}{y_i}|
$$

## WMAPE

$$
WMAPE=\frac{\sum_n|y^{'}-y|}{\sum_n{y}}
$$

## MASE

$$
MASE = \frac{1}{h}\sum_{t=1}^{h}(\frac{y_t' - y_t}{\frac{1}{n-1}\sum_{i=2}^{n}|y_i-y_{i-1}|})
$$

## RMSSE

$$
RMSSE = \sqrt{\frac{1}{h}\sum_{t=1}^{h}\frac{(y_t' - y_t)^2}{\frac{1}{n}\sum_{i=1}^{n}(y_i - y_{i-1})^2}}
$$

