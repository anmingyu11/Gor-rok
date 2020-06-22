> https://zhuanlan.zhihu.com/p/73330018

# 机器学习基础，回归模型评估指标

回归模型中常用的评估指标可以分如下几类：

1. MAE系列，即由Mean Absolute Error衍生得到的指标；

2. MSE系列，即由Mean Squared Error衍生得到的指标；

3. R²系列；

注：在英语中，error和deviation的含义是一样的，所以Mean Absolute Error也可以叫做Mean Absolute Deviation(MAD)，其他指标同理可得；

## 1. MAE系列

MAE 全称 Mean Absolute Error (平均绝对误差)。

设$N$为样本数量，$y_i$为实际值，$y_i'$为预测值，那么 MAE 的定义如下
$$
MAE = \frac{1}{N}\sum_{i=1}^{N} |y_i'-y_i|
$$
由 MAE 衍生可以得到：

Mean Absolute Pencentage Error (MAPE，平均绝对百分比误差)，相当于加权版的 MAE。

### MAPE

$$
MAPE = \frac{1}{N}\sum_{i=1}^{N}|\frac{y_i'-y_i}{y_i}|
$$

MAPE 可以看做是 MAE 和 MPE (Mean Percentage Error) 综合而成的指标
$$
MPE = \frac{100\%}{N}\sum_{i=1}^{N}\frac{y_i' - y_i}{y_i}
$$
从 MAPE 公式中可以看出有个明显的 bug——当实际值$y_i$为$0$时就会得到无穷大值(实际值$y_i$的绝对值$<1$也会过度放大误差)。为了避免这个 bug，MAPE一般用于实际值不会为$0$的情形。

Sungil Kima & Heeyoung Kim(2016) 提出 MAAPE(mean arctangent absolute percentage error) 方法，在保持 MAPE 的算法思想下克服了上面那个 bug

更多参考 A new metric of absolute percentage error for intermittent demand forecasts,Sungil Kima & Heeyoung Kim, 2016。
$$
MAAPE = \frac{1}{N}\sum_{i=1}^{N}arctan(|\frac{y_{i}^{'} -y_i}{y_i}|)
$$
考虑Absolute Error
$$
|y_i' - y_i|
$$
可能存在 Outlier 的情况，此时 Median Abosulte Error (MedAE, 中位数绝对误差)可能是更好的选择。
$$
MedAE = \mathop{median}\limits_{i=1,...,N}|y_i-y_i'|
$$

## 2. MSE系列

MSE全称Mean Squared Error(均方误差)，也可以称为Mean Squared Deviation . 

更多参考：[https://en.wikipedia.org/wiki/Mean_squared_error](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Mean_squared_error)
$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i'-y_i)^2
$$
由MSE可以衍生得到均方根误差(Root Mean Square Error, RMSE, 或者RMSD)
$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i'-y_i)^2}
$$
RMSE可以进行归一化(除以全距或者均值)从而得到归一化的均方根误差(Normalized Root Mean Square Error, NRMSE).

RMSE可以进行归一化(除以全距或者均值)从而得到归一化的均方根误差(Normalized Root Mean Square Error, NRMSE).
$$
NRMSE = \frac{RMSE}{ymax-ymin}
NRMSE = \frac{RMSE}{\bar y}
$$
RMSE还有其他变式：

1. RMSLE(Root Mean Square Logarithmic Error)

$$
RMSLE = \sqrt{\frac{1}{N}\sum_{i=1}^N(log(y_i + 1) - log(y_i' + 1))^2}
$$

2. RMSPE(Root Mean Square Percentage Error)
   $$
   RMSPE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}|\frac{y_i - y_i'}{y_i} |}
   $$

3. 对于数值序列出现长尾分布的情况，可以选择 MSLE(Mean squared logarithmic error，均方对数误差)，对原有数据取对数后再进行比较(公式中+1是为了避免数值为0时出现无穷值)。
   $$
   MSLE = \frac{1}{N}\sum_{i=1}^N(log(y_i + 1) - log(y_i' + 1))^2
   $$

## 3. R²系列

R²(R squared, Coefficient of determination)，中文翻译为“决定系数”或者“拟合优度”，反映的是预测值对实际值的解释程度。

注意：R²和相关系数的平方不是一回事(只在简单线性回归条件下成立)。
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\\= 1 - \frac{\sum_{i=1}^{N}(y_i - y_i')^2}{\sum_{i=1}^{N}(y_i-\bar y)^2}
$$
其中 总平方和$SS_{tot}$ = 回归平方和$SS_{reg}$ +残差平方和$SS_{res}$
$$
SS_{tot} = \sum_{i=1}^{N}(y_i - y_i')^2
$$

$$
SS_{reg} = \sum_{i=1}^{N}(y_i' - \bar y)^2
$$

$$
SS_{res} = \sum_{i=1}^{N}(y_i - \bar y)^2
$$

回归模型中，增加额外的变量会提升R²，但这种提升可能是虚假的，因此提出矫正的R²(Adjusted R²，符号表示为。

R方可以用来评价模型的拟合程度。当我们在评价拟合程度的同时，也考虑到模型的复杂程度，那么就是修正R方。

我们知道在其他变量不变的情况下，引入新的变量，总能提高模型的$R^2$。修正$R^2$就是相当于给变量的个数加惩罚项。

换句话说，如果两个模型，样本数一样，$R^2$一样，那么从修正$R^2$的角度看，使用变量个数少的那个模型更优。使用修正$R^2$也算一种[奥卡姆剃刀](http://sofasofa.io/forum_main_post.php?postid=1000306)的实例。
$$
R^2_{\text{Adj}}=1-\frac{SS_{\text{Res}}/(n-p-1)}{SS_{\text{Total}}/(n-1)}
\\=1-\frac{SS_{\text{Res}}}{SS_{\text{Total}}}\frac{(n-1)}{(n-p-1)}
\\=1-(1-R^2)\frac{n-p-1}{n-1}
$$
综上，在选用评价指标时，需要考虑

1. **数据中是否有0**，如果有0值就不能用MPE、MAPE之类的指标；

2. **数据的分布如何**，如果是长尾分布可以选择带对数变换的指标，中位数指标比平均数指标更好；

3. **是否存在极端值**，诸如MAE、MSE、RMSE之类容易受到极端值影响的指标就不要选用；

4. **得到的指标是否依赖于量纲**(即绝对度量，而不是相对度量)，如果指标依赖量纲那么不同模型之间可能因为量纲不同而无法比较；