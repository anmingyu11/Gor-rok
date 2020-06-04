## Prophet 简介

Facebook 去年开源了一个时间序列预测的算法，叫做 **fbprophet**，它的官方网址与基本介绍来自于以下几个网站：

1. Github：[https://github.com/facebook/prophet](https://link.zhihu.com/?target=https%3A//github.com/facebook/prophet)
2. 官方网址：[https://facebook.github.io/prophet/](https://link.zhihu.com/?target=https%3A//facebook.github.io/prophet/)
3. 论文名字与网址：Forecasting at scale，[https://peerj.com/preprints/3190/](https://link.zhihu.com/?target=https%3A//peerj.com/preprints/3190/)

从官网的介绍来看，Facebook 所提供的 prophet 算法不仅可以处理时间序列存在一些异常值的情况，也可以处理部分缺失值的情形，还能够几乎全自动地预测时间序列未来的走势。

从论文上的描述来看，这个 prophet 算法是**基于时间序列分解和机器学习的拟合**来做的，其中在拟合模型的时候使用了 pyStan 这个开源工具，因此能够在较快的时间内得到需要预测的结果。

除此之外，为了方便统计学家，机器学习从业者等人群的使用，prophet 同时提供了 R 语言和 Python 语言的接口。

从整体的介绍来看，如果是一般的商业分析或者数据分析的需求，都可以尝试使用这个开源算法来预测未来时间序列的走势。

## Prophet 的算法原理

### Prophet 数据的输入和输出

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/1.png)

首先让我们来看一个常见的时间序列场景，黑色表示原始的时间序列离散点，深蓝色的线表示使用时间序列来拟合所得到的取值，而浅蓝色的线表示时间序列的一个置信区间，也就是所谓的合理的上界和下界。prophet 所做的事情就是：

1. 输入已知的时间序列的时间戳和相应的值；
2. 输入需要预测的时间序列的长度；
3. 输出未来的时间序列走势。
4. 输出结果可以提供必要的统计指标，包括拟合曲线，上界和下界等。

就一般情况而言，时间序列的离线存储格式为时间戳和值这种格式，更多的话可以提供时间序列的 ID，标签等内容。

因此，离线存储的时间序列通常都是以下的形式。

其中 date 指的是具体的时间戳，category 指的是某条特定的时间序列 id，value 指的是在 date 下这个 category 时间序列的取值，label 指的是人工标记的标签（'0' 表示异常，'1‘ 表示正常，'unknown' 表示没有标记或者人工判断不清）。

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/2.png)

而 fbprophet 所需要的时间序列也是这种格式的，根据官网的描述，只要用 csv 文件存储两列即可，第一列的名字是 'ds', 第二列的名称是 'y'。第一列表示时间序列的时间戳，第二列表示时间序列的取值。通过 prophet 的计算，可以计算出 yhat，yhat_lower，yhat_upper，分别表示时间序列的预测值，预测值的下界，预测值的上界。两份表格如下面的两幅图表示。

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/3.png)

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/4.png)

### **Prophet 的算法实现**

在时间序列分析领域，有一种常见的分析方法叫做时间序列的分解（Decomposition of Time Series），

它把时间序列 $y_{t}$ 分成几个部分，分别是季节项 $S_{t}$，趋势项 $T_{t}$，剩余项 $R_{t}$。也就是说对所有的 $t \geq 0$，都有
$$
y_{t} = S_{t} + T_{t} + R_{t}
$$
除了加法的形式，还有乘法的形式，也就是：
$$
y_{t} = S_{t} \times T_{t} \times R_{t}.
$$
以上式子等价于
$$
\ln y_{t} = \ln S_{t} + \ln T_{t} + \ln R_{t}
$$
所以，有的时候在预测模型的时候，会先取对数，然后再进行时间序列的分解，就能得到乘法的形式。在 fbprophet 算法中，作者们基于这种方法进行了必要的改进和优化。

一般来说，在实际生活和生产环节中，除了季节项，趋势项，剩余项之外，通常还有节假日的效应。所以，在 prophet 算法里面，作者同时考虑了以上四项，也就是：
$$
y(t) = g(t) + s(t) + h(t) + \epsilon_{t}.
$$
 $g(t)$ 表示趋势项，它表示时间序列在非周期上面的变化趋势；

$s(t)$ 表示周期项，或者称为季节项，一般来说是以周或者年为单位；

$h(t)$示节假日项，表示在当天是否存在节假日；

$\epsilon_{t}$ 表示误差项或者称为剩余项。

Prophet 算法就是通过拟合这几项，然后最后把它们累加起来就得到了时间序列的预测值。

#### 趋势项模型$g(t)$

在 Prophet 算法里面，趋势项有两个重要的函数，一个是基于逻辑回归函数（logistic function）的，另一个是基于分段线性函数（piecewise linear function）的。

首先，我们来介绍一下基于逻辑回归的趋势项是怎么做的。

如果回顾逻辑回归函数的话，一般都会想起这样的形式：
$$
\sigma(x) = 1/(1+e^{-x})
$$
它的导数是
$$
\sigma'(x) = \sigma(x) \cdot(1-\sigma(x))
$$
并且 
$$
\lim_{x\rightarrow +\infty} \sigma(x) = 1
$$
如果增加一些参数的话，那么逻辑回归就可以改写成：
$$
f(x) = C / (1 + e^{-k(x-m)})
$$
这里的 $C$ 称为曲线的最大渐近值， $k$ 表示曲线的增长率，$m$ 表示曲线的中点。

当 $C=1, k = 1, m =0$ 时，恰好就是大家常见的 sigmoid 函数的形式。

从 sigmoid 的函数表达式来看，它满足以下的微分方程：
$$
y'=y(1-y)
$$

----------



那么，如果使用分离变量法来求解微分方程$y'=y(1-y)$就可以得到:
$$
\frac{y'}{y} + \frac{y'}{1-y} = 1 \Rightarrow \ln\frac{y}{1-y} = 1 \Rightarrow y = 1/(1+K e^{-x})
$$
但是在现实环境中，函数$f(x) = C / (1+e^{-k(x-m)})$的三个参数 $C, k, m$ 不可能都是常数，而很有可能是随着时间的迁移而变化的，因此，在 Prophet 里面，作者考虑把这三个参数全部换成了随着时间而变化的函数，也就是 
$$
C = C(t), k = k(t), m = m(t)
$$


除此之外，在现实的时间序列中，曲线的走势肯定不会一直保持不变，在某些特定的时候或者有着某种潜在的周期曲线会发生变化，这种时候，就有学者会去研究变点检测，也就是所谓 change point detection。例如下面的这幅图的 $t_{1}^{*}, t_{2}^{*}$ 就是时间序列的两个变点。

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/5.png)

在 Prophet 里面，是需要设置变点的位置的，而每一段的趋势和走势也是会根据变点的情况而改变的。

在程序里面有两种方法，一种是通过人工指定的方式指定变点的位置；另外一种是通过算法来自动选择。

在默认的函数里面，Prophet 会选择`n_changepoints = 25` 个变点，然后设置变点的范围是前 `80%`（changepoint_range），也就是在时间序列的前 `80%` 的区间内会设置变点。

通过` forecaster.py` 里面的` set_changepoints` 函数可以知道，首先要看一些边界条件是否合理，例如时间序列的点数是否少于` n_changepoints` 等内容；

其次如果边界条件符合，那变点的位置就是均匀分布的，这一点可以通过 `np.linspace` 这个函数看出来。

-------------------

下面假设已经放置了 $S$ 个变点了，并且变点的位置是在时间戳 $s_{j}, 1\leq j\leq S$ 上，那么在这些时间戳上，我们就需要给出增长率的变化，也就是在时间戳 $s_{j}$ 上发生的 `change in rate`。

可以假设有这样一个向量：$\boldsymbol{\delta}\in\mathbb{R}^{S},$ 其中 $\delta_{j}$ 表示在时间戳 $s_{j}$ 上的增长率的变化量。

如果一开始的增长率我们使用 $k$ 来代替的话，那么在时间戳$t$上的增长率就是 $k + \sum_{j:t>s_{j}} \delta_{j}$，通过一个指示函数 $\mathbf{a}(t)\in \{0,1\}^{S}$ 就是
$$
a_{j}(t) = \begin{cases} 1, \text{ if } t\geq s_{j},\\ 0, \text{ otherwise.} \end{cases}
$$
那么在时间戳 $t$ 上面的增长率就是 $k + \mathbf{a}^{T}\boldsymbol{\delta}.$ 一旦变化量 $k$ 确定了，另外一个参数 $m$ 也要随之确定。在这里需要把线段的边界处理好，因此通过数学计算可以得到：
$$
\gamma_{j} = \bigg(s_{j} - m - \sum_{\ell <j} \gamma_{\ell} \bigg) \cdot \bigg( 1- \frac{k + \sum_{\ell < j} \delta_{\ell}}{k + \sum_{\ell\leq j}\delta_{\ell}} \bigg)
$$
所以，分段的逻辑回归增长模型就是：
$$
g(t) = \frac{C(t)}{1+exp(-(k+\boldsymbol{a}(t)^{t}\boldsymbol{\delta}) \cdot (t - (m+\boldsymbol{a}(t)^{T}\boldsymbol{\gamma})},
$$
其中，
$$
\boldsymbol{a}(t) = (a_{1}(t),\cdots,a_{S}(t))^{T}, \boldsymbol{\delta} = (\delta_{1},\cdots,\delta_{S})^{T}, \boldsymbol{\gamma} = (\gamma_{1},\cdots,\gamma_{S})^{T}.
$$
在逻辑回归函数里面，有一个参数是需要提前设置的，那就是 Capacity，也就是所谓的 $C(t) $，在使用 Prophet 的 `growth = ‘logistic’ `的时候，需要提前设置好  $C(t) $ 的取值才行。

再次，我们来介绍一下基于分段线性函数的趋势项是怎么做的。众所周知，线性函数指的是 $y = kx + b$ 而分段线性函数指的是在每一个子区间上，函数都是线性函数，但是在整段区间上，函数并不完全是线性的。正如下图所示，分段线性函数就是一个折线的形状。

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/6.png)

因此，基于分段线性函数的模型形如：
$$
g(t)=(k+\boldsymbol{a}(t)\boldsymbol{\delta})\cdot t+(m+\boldsymbol{a}(t)^{T}\boldsymbol{\gamma}),
$$
其中 $k$ 表示增长率（growth rate），$\boldsymbol{\delta}$ 表示增长率的变化量，$m$ 表示 offset parameter。而这两种方法（分段线性函数与逻辑回归函数）最大的区别就是 $\boldsymbol{\gamma} $ 的设置不一样，在分段线性函数中，
$$
\boldsymbol{\gamma}=(\gamma_{1},\cdots,\gamma_{S})^{T},
\gamma_{j}=-s_{j}\delta_{j}.
$$
注意：这与之前逻辑回归函数中的设置是不一样的。

在 prophet 的源代码中，`forecast.py` 这个函数里面包含了最关键的步骤，其中 `piecewise_logistic` 函数表示了前面所说的基于逻辑回归的增长函数，它的输入包含了 `cap` 这个指标，因此需要用户事先指定 `capacity`。

而在 `piecewise_linear` 这个函数中，是不需要 `capacity` 这个指标的，因此 `m = Prophet()` 这个函数默认的使用`growth = ‘linear’` 这个增长函数，也可以写作 `m = Prophet(growth = ‘linear’)`；

如果想用 `growth = ‘logistic’`，就要这样写：

```python
m = Prophet(growth='logistic')
df['cap'] = 6
m.fit(df)
future = m.make_future_dataframe(periods=prediction_length, freq='min')
future['cap'] = 6
```

#### 变点的选择（Changepoint Selection）

在介绍变点之前，先要介绍一下 Laplace 分布，它的概率密度函数为：
$$
f(x|\mu, b) = exp\bigg(-|x-\mu|/b\bigg)/2b,
$$
其中 $\mu$ 表示位置参数，$b>0$ 表示尺度参数。Laplace 分布与正态分布有一定的差异。

在 Prophet 算法中，是需要给出变点的位置，个数，以及增长的变化率的。

因此，有三个比较重要的指标，那就是

1. changepoint_range，
2. n_changepoint，
3. changepoint_prior_scale。

`changepoint_range` 指的是百分比，需要在前 `changepoint_range` 那么长的时间序列中设置变点，在默认的函数中是 `changepoint_range = 0.8`。`n_changepoint` 表示变点的个数，在默认的函数中是 `n_changepoint = 25`。`changepoint_prior_scale` 表示变点增长率的分布情况，在论文中， $\delta_{j} \sim Laplace(0,\tau)$，这里的 $\tau$ 就是` change_point_scale`。

在整个开源框架里面，在默认的场景下，变点的选择是基于时间序列的前 80% 的历史数据，然后通过等分的方法找到 25 个变点（change points），而变点的增长率是满足 Laplace 分布 $\delta_{j} \sim Laplace (0,0.05)$ 的。

因此，当$\tau$趋近于零的时候，$\delta_{j}$也是趋向于零的，此时的增长函数将变成全段的逻辑回归函数或者线性函数。

这一点从 $g(t)$ 的定义可以轻易地看出。

#### 对未来的预估(Trend Forecast Uncertainty)

从历史上长度为 $T$ 的数据中，我们可以选择出 $S$ 个变点，它们所对应的增长率的变化量是  $\delta_{j} \sim Laplace(0,\tau)$ 。此时我们需要预测未来，因此也需要设置相应的变点的位置，从代码中看，在 `forecaster.py` 的 `sample_predictive_trend` 函数中，通过 `Poisson` 分布等概率分布方法找到新增的 `changepoint_ts_new` 的位置，然后与` changepoint_t` 拼接在一起就得到了整段序列的 `changepoint_ts`。

```python
changepoint_ts_new = 1 + np.random.rand(n_changes) * (T - 1)
changepoint_ts = np.concatenate((self.changepoints_t,changepoint_ts_new))
```

第一行代码的 1 保证了 changepoint_ts_new 里面的元素都大于 change_ts 里面的元素。除了变点的位置之外，也需要考虑 $\delta$ 的情况。这里令 $\lambda = \sum_{j=1}^{S}|\delta_{j}|/S$ ，于是新的增长率的变化量就是按照下面的规则来选择的：当 $j>T$ 时
$$
\delta_{j}=\begin{cases} 0 \text{, with probability } (T-S)/T \\ \sim Laplace(0,\lambda) \text{, with probability } S/T \end{cases}.
$$

### **季节性趋势**

几乎所有的时间序列预测模型都会考虑这个因素，因为时间序列通常会随着天，周，月，年等季节性的变化而呈现季节性的变化，也称为周期性的变化。

对于周期函数而言，大家能够马上联想到的就是正弦余弦函数。而在数学分析中，区间内的周期性函数是可以通过正弦和余弦的函数来表示的：假设 $f(x)$ 是以 $2\pi$ 为周期的函数，那么它的傅立叶级数就是 $a_{0} + \sum_{n=1}^{\infty}(a_{n}\cos(nx) + b_{n}\sin(nx))$。

在论文中，作者使用傅立叶级数来模拟时间序列的周期性。假设 $P$ 表示时间序列的周期，$P = 365.25$ 表示以年为周期，$P = 7$ 表示以周为周期。它的傅立叶级数的形式都是：
$$
s(t) = \sum_{n=1}^{N}\bigg( a_{n}\cos\bigg(\frac{2\pi n t}{P}\bigg) + b_{n}\sin\bigg(\frac{2\pi n t}{P}\bigg)\bigg).
$$
就作者的经验而言，对于以年为周期的序列（$P = 365.25$）而言，$N = 10$；对于以周为周期的序列（$P = 7$）而言，$N = 3$。这里的参数可以形成列向量：
$$
\boldsymbol{\beta} = (a_{1},b_{1},\cdots,a_{N},b_{N})^{T}
$$
当 $N = 10$ 时，
$$
X(t) = \bigg[\cos(\frac{2\pi(1)t}{365.25}),\cdots,\sin(\frac{2\pi(10)t}{365.25})\bigg]
$$
当 $N = 3$ 时，
$$
X(t) = \bigg[\cos(\frac{2\pi(1)t}{7}),\cdots,\sin(\frac{2\pi(3)t}{7})\bigg]
$$
因此，时间序列的季节项就是：$s(t) = X(t) \boldsymbol{\beta}$ , 而 $\boldsymbol{\beta}$ 的初始化是 $\boldsymbol{\beta} \sim Normal(0,\sigma^{2})$。

这里的 $\sigma$ 是通过 `seasonality_prior_scale` 来控制的，也就是说 $\sigma=$ seasonality_prior_scale。

这个值越大，表示季节的效应越明显；这个值越小，表示季节的效应越不明显。同时，在代码里面，`seasonality_mode` 也对应着两种模式，分别是加法和乘法，默认是加法的形式。在开源代码中，$X(t)$ 函数是通过 `fourier_series` 来构建的。

#### 节假日效应(holidays and events)

在现实环境中，除了周末，同样有很多节假日，而且不同的国家有着不同的假期。在 Prophet 里面，通过维基百科里面对各个国家的节假日的描述，`hdays.py` 收集了各个国家的特殊节假日。除了节假日之外，用户还可以根据自身的情况来设置必要的假期，例如 The Super Bowl，双十一等。

![](https://zr9558.files.wordpress.com/2018/11/prophetholiday1.png)

由于每个节假日对时间序列的影响程度不一样，例如春节，国庆节则是七天的假期，对于劳动节等假期来说则假日较短。因此，不同的节假日可以看成相互独立的模型，并且可以为不同的节假日设置不同的前后窗口值，表示该节假日会影响前后一段时间的时间序列。用数学语言来说，对与第 $i$ 个节假日来说， $D_{i}$ 表示该节假日的前后一段时间。为了表示节假日效应，我们需要一个相应的指示函数（indicator function），同时需要一个参数 $\kappa_{i}$ 来表示节假日的影响范围。

假设我们有 $L$ 个节假日，那么
$$
h(t)=Z(t) \boldsymbol{\kappa}=\sum_{i=1}^{L} \kappa_{i}\cdot 1_{\{t\in D_{i}\}},
$$
其中 $Z(t)=(1_{\{t\in D_{1}\}},\cdots,1_{\{t\in D_{L}\}})$ 和 $\boldsymbol{\kappa}=(\kappa_{1},\cdots,\kappa_{L})^{T}]$

其中 $\boldsymbol{\kappa}\sim Normal(0,v^{2})$ 并且该正态分布是受到 $v =$ holidays_prior_scale 这个指标影响的。默认值是 10，当值越大时，表示节假日对模型的影响越大；当值越小时，表示节假日对模型的效果越小。用户可以根据自己的情况自行调整。

### **模型拟合（Model Fitting）**

按照以上的解释，我们的时间序列已经可以通过增长项，季节项，节假日项来构建了，i.e.
$$
y(t)=g(t)+s(t)+h(t)+\epsilon
$$
下一步我们只需要拟合函数就可以了，在 Prophet 里面，作者使用了 pyStan 这个开源工具中的 L-BFGS 方法来进行函数的拟合。具体可以参考 forecast.py 里面的 stan_init 函数。

### Prophet 中可以设置的参数

在 Prophet 中，用户一般可以设置以下四种参数：

1. Capacity：在增量函数是逻辑回归函数的时候，需要设置的容量值。
2. Change Points：可以通过 `n_changepoints` 和 `changepoint_range` 来进行等距的变点设置，也可以通过人工设置的方式来指定时间序列的变点。
3. 季节性和节假日：可以根据实际的业务需求来指定相应的节假日。
4. 光滑参数：$\tau=$ changepoint_prior_scale 可以用来控制趋势的灵活度，$\sigma=$ seasonality_prior_scale 用来控制季节项的灵活度，$v=$ holidays prior scale 用来控制节假日的灵活度。

如果不想设置的话，使用 Prophet 默认的参数即可。

### **Prophet 的实际使用**

#### **Prophet 的简单使用**

因为 Prophet 所需要的两列名称是 ‘ds’ 和 ‘y’，其中，’ds’ 表示时间戳，’y’ 表示时间序列的值，因此通常来说都需要修改 pd.dataframe 的列名字。如果原来的两列名字是 ‘timestamp’ 和 ‘value’ 的话，只需要这样写：

```python
df = df.rename(columns={'timestamp':'ds', 'value':'y'})
```

如果 ‘timestamp’ 是使用 unixtime 来记录的，需要修改成 YYYY-MM-DD hh:mm:ss 的形式：

```python
df['ds'] = pd.to_datetime(df['ds'],unit='s')
```

在一般情况下，时间序列需要进行归一化的操作，而 pd.dataframe 的归一化操作也十分简单：

```python
df['y'] = (df['y'] - df['y'].mean()) / (df['y'].std())
```

然后就可以初始化模型，然后拟合模型，并且进行时间序列的预测了。

```python
#初始化模型：m = Prophet()
#拟合模型：m.fit(df)
#计算预测值：periods 表示需要预测的点数，freq 表示时间序列的频率。
future = m.make_future_dataframe(periods=30, freq='min')
future.tail()
forecast = m.predict(future)
```

而 freq 指的是 pd.dataframe 里面的一个指标，’min’ 表示按分钟来收集的时间序列。具体参见文档：http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/7.png)

在进行了预测操作之后，通常都希望把时间序列的预测趋势画出来：

```python
# 画出预测图：
m.plot(forecast)
# 画出时间序列的分量：
m.plot_components(forecast)
```

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/8.png)

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/9.png)

如果要画出更详细的指标，例如中间线，上下界，那么可以这样写：

```python
x1 = forecast['ds']
y1 = forecast['yhat']
y2 = forecast['yhat_lower']
y3 = forecast['yhat_upper']
plt.plot(x1,y1)
plt.plot(x1,y2)
plt.plot(x1,y3)
plt.show()
```

![](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/10.png)

其实 Prophet 预测的结果都放在了变量 forecast 里面，打印结果的话可以这样写：第一行是打印所有时间戳的预测结果，第二行是打印最后五个时间戳的预测结果。

```python
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
```

### **Prophet 的参数设置**

Prophet 的默认参数可以在 forecaster.py 中看到：

```python
def __init__(
    self,
    growth='linear',
    changepoints=None,
    n_changepoints=25, 
    changepoint_range=0.8,
    yearly_seasonality='auto',
    weekly_seasonality='auto',
    daily_seasonality='auto',
    holidays=None,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000,
):
```

#### **增长函数的设置**

在 Prophet 里面，有两个增长函数，分别是分段线性函数（linear）和逻辑回归函数（logistic）。而 m = Prophet() 默认使用的是分段线性函数（linear），并且如果要是用逻辑回归函数的时候，需要设置 capacity 的值，i.e. df[‘cap’] = 100，否则会出错。

```
m = Prophet()
m = Prophet(growth='linear')
m = Prophet(growth='logistic')
```

#### **变点的设置**

在 Prophet 里面，变点默认的选择方法是前 80% 的点中等距选择 25 个点作为变点，也可以通过以下方法来自行设置变点，甚至可以人为设置某些点。

```
m = Prophet(n_changepoints=25)
m = Prophet(changepoint_range=0.8)
m = Prophet(changepoint_prior_scale=0.05)
m = Prophet(changepoints=['2014-01-01'])
```

而变点的作图可以使用：

```
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

![prophetexample8](https://zr9558.files.wordpress.com/2018/11/prophetexample8.png?w=474)

#### **周期性的设置**

通常来说，可以在 Prophet 里面设置周期性，无论是按月还是周其实都是可以设置的，例如：

```python
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5,fourier_order=5)
m = Prophet(weekly_seasonality=True)
m.add_seasonality(name='weekly',period=7,fourier_order=3,prior_scale=0.1)
```

![prophetexample9](/Users/helloword/Anmingyu/Gor-rok/TimeSeries/Prophet/Facebook时间序列预测算法Prophet的研究/11.png)

#### **节假日的设置**

有的时候，由于双十一或者一些特殊节假日，我们可以设置某些天数是节假日，并且设置它的前后影响范围，也就是 lower_window 和 upper_window。

```python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})

superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})

holidays = pd.concat((playoffs, superbowls))

m = Prophet(holidays=holidays, holidays_prior_scale=10.0)
```

## **结束语**

对于商业分析等领域的时间序列，Prophet 可以进行很好的拟合和预测，但是对于一些周期性或者趋势性不是很强的时间序列，用 Prophet 可能就不合适了。但是，Prophet 提供了一种时序预测的方法，在用户不是很懂时间序列的前提下都可以使用这个工具得到一个能接受的结果。具体是否用 Prophet 则需要根据具体的时间序列来确定。

## **参考文献：**

1. https://otexts.org/fpp2/components.html
2. https://en.wikipedia.org/wiki/Decomposition_of_time_series
3. A review of change point detection methods, CTruong, L. Oudre, N.Vayatis
4. https://github.com/facebook/prophet
5. https://facebook.github.io/prophet/