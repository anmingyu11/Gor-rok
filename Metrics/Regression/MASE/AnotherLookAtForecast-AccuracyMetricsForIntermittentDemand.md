# ANOTHER LOOK AT FORECAST-ACCURACY METRICS FOR INTERMITTENT DEMAND

> 另一个关于 intermittent-demand的预测准确性指标

Rob Hyndman is Professor of Statistics at Monash University, Australia, and Editor in Chief of the International Journal of Forecasting. He is an experienced consultant who has worked with over 200 clients during the last 20 years, on projects covering all areas of applied statistics, from forecasting to the ecology of lemmings. He is coauthor of the well-known textbook, Forecasting: Methods and Applications (Wiley, 1998), and he has published more than 40 journal articles. Rob is Director of the Business and Economic Forecasting Unit, Monash University, one of the leading forecasting research groups in the world.

> 罗伯·海恩德曼（Rob Hyndman）是澳大利亚莫纳什大学（Monash University）的统计学教授，也是《国际预测杂志》的主编。 他是一位经验丰富的顾问，在过去的20年中，他与200多个客户合作过，涉及从预测到旅鼠生态学等所有应用统计领域的项目。 他是著名教科书《预测：方法和应用》（Wiley，1998年）的合著者，并发表了40多篇期刊文章。 罗伯（Rob）是莫纳什大学（Monash University）商业和经济预测部门的主管，莫纳什大学是世界领先的预测研究小组之一。

Preview: Some traditional measurements of forecast accuracy are unsuitable for intermittent-demand data because they can give infinite or undefined values. Rob Hyndman summarizes these forecast accuracy metrics and explains their potential failings. He also introduces a new metric—the mean absolute scaled error (MASE)—which is more appropriate for intermittent-demand data. More generally, he believes that the MASE should become the standard metric for comparing forecast accuracy across multiple time series.

> 预览：一些传统的预测准确性度量不适合intermittent-demand数据，因为它们可以给出infinite或 undefined 的值。 Rob Hyndman总结了这些预测准确性指标并解释了它们的潜在缺陷。 他还介绍了一种新的度量标准-平均绝对比例误差（MASE），它更适合于 intermittent-demand数据。 更普遍地说，他认为MASE应该成为比较多个时间序列的预测准确性的标准指标。

- There are four types of forecast-error metrics: scale-dependent metrics such as the mean absolute error (MAE or MAD); percentage-error metrics such as the mean absolute percent error (MAPE); relative-error metrics, which average the ratios of the errors from a designated method to the errors of a naive method; and scale-free error metrics, which express each error as a ratio to an average error from a baseline method. 
- For assessing accuracy on a single series, I prefer the MAE because it is easiest to understand and compute. However, it cannot be compared across series because it is scale dependent; it makes no sense to compare accuracy on different scales.
- Percentage errors have the advantage of being scale independent, so they are frequently used to compare forecast performance between different data series. But measurements based on percentage errors have the disadvantage of being infinite or undefined if there are zero values in a series, as is frequent for intermittent data.
- Relative-error metrics are also scale independent. However, when the errors are small, as they can be with intermittent series, use of the naïve method as a benchmark is no longer possible because it would involve division by zero.
- The scale-free error metric I call the mean absolute scaled error (MASE) can be used to compare forecast methods on a single series and also to compare forecast accuracy between series. This metric is well suited to intermittent-demand series because it never gives infinite or undefined values.



> - 有四种类型的预测误差指标:依赖于规模的指标，如平均绝对误差(MAE或MAD);百分比误差指标，如平均绝对百分比误差(MAPE);相对误差度量，将指定方法的误差与朴素方法的误差的比值平均;和无标度错误度量，它将每个错误表示为与基线方法的平均错误的比率。
> - 为了评估单个系列的准确性，我更喜欢MAE，因为它最容易理解和计算。 但是，由于它与比例有关，因此无法在系列中进行比较。 比较不同规模的准确性没有任何意义。
> - 百分比误差具有规模无关的优点，因此经常用于比较不同数据序列之间的预测性能。但是基于百分比误差的测量有一个缺点，那就是如果一个序列中有0个值，那么测量值就是无限的，或者是没有定义的，这对于intermittent-demand来说是很常见的。
> - 相对误差指标也与规模无关。 但是，当误差较小时（如间歇序列的误差一样），就不再可能使用朴素方法作为基准，因为它将涉及被零除。
> - the scale-free error metric，我称之为平均绝对标度误差(MASE)，可以用来比较单个Series的预测方法，也可以比较Series之间的预测精度。这个度量非常适合于ntermittent-demand序列，因为它从不给出无穷大或未定义的值。

## **Introduction: Three Ways to Generate Forecasts**

>  ## 生成预测结果的三种方法

There are three ways we may generate forecasts (F) of a quantity (Y) from a particular forecasting method:

1. We can compute forecasts from a common origin $t$ (for example, the most recent month) for a sequence of forecast horizons $F_{n+1},...,F_{n+m}$ based on data from times $t = 1,...,n.$ This is the standard procedure implemented by forecasters in real time.
2. We can vary the origin from which forecasts are made but maintain a consistent forecast horizon.  For example, we can generate a series of one-period-ahead forecasts F_{1+h},...,F_{m+h} where each F_{j+h} is based on data from times t = 1,..., *j .* This procedure is done not only to give attention to the forecast errors at a particular horizon but also to show how the forecast error changes as the horizon lengthens.
3. We may generate forecasts for a single future period using multiple data series, such as a collection of products or items. This procedure can be useful to demand planners as they assess aggregate accuracy over items or products at a location. This is also the procedure that underlies forecasting competitions, which compare the accuracy of different methods across multiple series.



> 我们可以用三种方法从一种特定的预测方法中得出某一数量(Y)的预测(F):
>
> 1. 我们可以从一个共同的起点 $t$ (例如，最近的一个月)来计算一系列的预测范围$F_{n+1}，…，F_{n+m}$，基于来自时间 $t = 1，…，n$的数据。​这是 forecaster 实现的标准程序 in real time。
> 2. 我们可以改变预测的来源，但要保持一致的 forcast horizon 。例如，我们可以生成一系列提前一个周期的预测$F_{1+h}，…，F_{m+h}$，其中每个$F_{1+h}$基于来自时间$t = 1,...,j$的数据.这个过程不仅是为了关注在一个particular horizon上的预测误差，而且是为了显示预测误差是如何随着 horizon 的延长而变化的。
> 3. 我们可以使用多个 Series (例如产品或项目的集合)来生成对单个未来时期的预测。当计划人员评估某一地点的项目或产品的总体准确性时，这一程序对他们来说是有用的。这也是预测比赛的基础程序，即比较多个Series中不同方法的准确性。

While these are very different situations, measuring forecast accuracy is similar in each case. It is useful to have a forecast accuracy metric that can be used for all three cases.

> 虽然这是非常不同的情况，但在每一种情况下测量预测精度是相似的。有一个可用于所有三种情况的预测准确度度量是有用的。

## **An Example of What Can Go Wrong**

> ## 可能出错的例子

Consider the classic intermittent-demand series shown in Figure 1. These data were part of a consulting project I did for a major Australian lubricant manufacturer.

> 考虑 Figure 1 中所示的典型 intermittent-demand 需求系列。这些数据是我为澳大利亚一家大型润滑油制造商做的一个咨询项目的一部分。

![Figure 1](/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MASE/Fig1.png)

Suppose we are interested in comparing the forecast accuracy of four simple methods: (1) the historical mean, using data up to the most recent observation; (2) the *naïve* or random-walk method, in which the forecast for each future period is the actual value for this period; (3) simple exponential smoothing; and (4) Croston’s method for intermittent demands (Boylan, 2005). For methods (3) and (4) I have used a smoothing parameter of 0.1.

> 假设我们有兴趣比较四种简单方法的预测精度:
>
> (1)使用最新观测数据的历史平均值;
>
> (2)朴素或随机游走法，即未来每个时段的预测为该时段的实际值;
>
> (3)简单指数平滑;
>
> (4)克罗斯顿间歇需求法(Boylan, 2005)。对于方法(3)和(4)，我使用了0.1的平滑参数。

I compared the *in-sample* performance of these methods by varying the origin and generating a sequence of one-period-ahead forecasts – the second forecasting procedure described in the introduction. I also calculated the *out-of-sample* performance based on forecasting the data in the hold-out period, using information from the fitting period alone. These out-of-sample forecasts are from one to twelve steps ahead and are not updated in the hold-out period.

> 我比较了这些方法的 in-sample 性能，方法是改变 origin 并生成一系列提前期的预测，这是引言中介绍的第二种预测程序。 我还基于仅在拟合期内的信息来预测hold-out period的数据，从而计算出“out-of-sample”性能。 这些样本外的预测要提前1到12步，并且在 hold-out period内不会更新。

Table 1 shows some commonly used forecast-accuracy metrics applied to these data. The metrics are all defined in the next section. There are many infinite values occurring in Table 1. These are caused by division by zero. The undefined values for the naïve method arise from the division of zero by zero. The only measurement that always gives sensible results for all four of the forecasting methods is the MASE, or the mean absolute scaled error. Infinite, undefined, or zero values plague the other accuracy measurements.

> 表1显示了应用于这些数据的一些常用的预测准确度指标。度量标准将在下一节中定义。表1中出现了许多无穷值。这些是由除 0 引起的。naive方法的未定义值来自零除以零。对于所有四种预测方法，唯一能给出合理结果的是MASE，即平均绝对尺度误差。无穷、未定义或零值困扰着其他metrics。

In this particular series, the out-of-sample period has smaller errors (is more predictable) than the in-sample period because the in-sample period includes some relatively large observations. In general, we would expect out-of-sample errors to be larger.

> 在此particular series中，由于 in-sample period 包含一些相对较大的观察值，因此 out-of-sample period 的误差（更容易预测）比 in-sample period 小。 通常，我们 expected out-of-sample 误差会更大。

![Table 1](/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MASE/table1.png)

## Measurement of Forecast Errors

We can measure and average forecast errors in several ways:

> 我们可以通过以下几种方式测量和平均预测误差:

## Scale-dependent errors

> 依赖尺度的误差

The forecast error is simply, $e_t = Y_t – F_t$ , regardless of how the forecast was produced. This is on the same scale as the data, applying to anything from ships to screws. Accuracy measurements based on $e_t$ are therefore scale-dependent.

> 不管预测是如何产生的，预测误差就是$ e_t = Y_t – F_t $。 这与数据的量纲相同，适用于从船舶到螺丝钉（从大到小）的所有内容。 因此，基于$ e_t $的精度测量值取决于量纲。

The most commonly used scale-dependent metrics are based on absolute errors or on squared errors:

Mean Absolute Error (MAE) = $mean(|e_t|) $

Geometric Mean Absolute Error (GMAE) = $gmean(|e_t|)$

Mean Square Error (MSE) = $mean(e^2_t)$

where “gmean” is a geometric mean.

> 最常用的与比例相关的度量基于绝对误差或平方误差：
>
> Mean Absolute Error (MAE) = $mean(|e_t|) $
>
> Geometric Mean Absolute Error (GMAE) = $gmean(|e_t|)$
>
> Mean Square Error (MSE) = $mean(e^2_t)$
>
> 其中“ gmean”是几何平均值。

The MAE is often abbreviated as the MAD (“D” for “deviation”). The use of absolute values or squared values prevents negative and positive errors from offsetting each other.

> MAE通常被缩写为MAD(“D”代表“偏差”)。绝对值或平方值的使用可以防止正负误差相互抵消。

Since all of these metrics are on the same scale as the data, none of them are meaningful for assessing a method’s accuracy across multiple series.

> 由于所有这些度量指标都与数据在同一个尺度上，所以它们对于多个 Series 评估方法的准确性都没有意义。

For assessing accuracy on a single series, I prefer the MAE because it is easiest to understand and compute. However, it cannot be compared between series because it is scale dependent.

> 为了评估单个Series 的准确性，我更喜欢MAE，因为它最容易理解和计算。 但是，由于它与量纲有关，因此无法在Series之间进行比较。

For intermittent-demand data, Syntetos and Boylan (2005) recommend the use of GMAE, although they call it the GRMSE. (The GMAE and GRMSE are identical because the square root and the square cancel each other in a geometric mean.) Boylan and Syntetos (this issue) point out that the GMAE has the flaw of being equal to zero when any error is zero, a problem which will occur when both the actual and forecasted demands are zero. This is the result seen in Table 1 for the naïve method.

> 对于 intermittent-demand 数据，Syntetos和Boylan（2005）建议使用GMAE，尽管他们将其称为 GRMSE 。（GMAE和GRMSE是相同的，因为平方根和平方在几何平均值上互相抵消。）Boylan和Syntetos（此问题）指出，当任何误差为零时，GMAE的缺点是等于零。 当实际需求和预测需求均为零时将发生的问题。 这是表1中朴素方法的结果。

Boylan and Syntetos claim that such a situation would occur only if an inappropriate forecasting method is used. However, it is not clear that the naïve method is always inappropriate. Further, Hoover indicates that division-by- zero errors in intermittent series are expected occurrences for repair parts. I suggest that the GMAE is problematic for assessing accuracy on intermittent-demand data.

> Boylan和Syntetos声称，只有在使用不适当的预测方法时才会出现这种情况。 但是，尚不清楚naive的方法是否总是不合适。此外，Hoover还指出，对于 repair parts，intermittent sereis 中出现的零分错误是可以预料的。我认为GMAE在评估 intermittend-demand 数据的准确性方面存在问题。

## **Percentage errors**

> 百分比误差

The percentage error is given by $p_t = 100e_t / Y_t$. Percentage errors have the advantage of being scale independent, so they are frequently used to compare forecast performance between different data series. The most commonly used metric is

Mean Absolute Percentage Error (MAPE) = $mean(|p_t|)$ 

Measurements based on percentage errors have the disadvantage of being infinite or undefined if there are zero values in a series, as is frequent for intermittent data. Moreover, percentage errors can have an extremely skewed distribution when actual values are close to zero. With intermittent-demand data, it is impossible to use the MAPE because of the occurrences of zero periods of demand.

> 百分比误差由$p_t = 100e_t / Y_t$给出。百分比误差具有规模无关的优点，因此常用于比较不同数据序列之间的预测性能。最常用的度量标准是
>
> 平均绝对误差百分比(MAPE) = $mean(|p_t|)$
>
> 如果在一个序列中有 0 值，那么基于百分比误差的测量缺点是会出现infinity或undefined，这在 inetrmittent-data 中很常见。此外，当实际值接近于零时，百分比误差可能具有极端倾斜的分布。对于 intermittent-data，不可能使用 MAPE ，因为出现了 0 peroids。

The MAPE has another disadvantage: it puts a heavier penalty on positive errors than on negative errors. This observation has led to the use of the “symmetric” MAPE (sMAPE) in the M3-competition (Makridakis & Hibon, 2000). It is defined by

sMAPE = $mean(200|Y_t - F_t | / (Y_t + F_t ))$

However, if the actual value $Y_t$ is zero, the forecast $F_t$ is likely to be close to zero. Thus the measurement will still involve division by a number close to zero. Also,the value of sMAPE can be negative, giving it an ambiguous interpretation.

> MAPE的另一个缺点是：对正错误的惩罚要比对负错误的惩罚更大。 这一观察结果导致在M3竞赛中使用“对称” MAPE（sMAPE）（Makridakis＆Hibon，2000）。 它的定义是
>
> sMAPE = $mean(200|Y_t - F_t | / (Y_t + F_t ))$
>
> 但是，如果实际值$ Y_t $为零，则预测$ F_t $可能接近零。 因此，测量仍将涉及除以接近零的数字。 而且，sMAPE的值可以为负，从而造成模棱两可的解释。

## Relative errors

An alternative to percentages for the calculation of scale-independent measurements involves dividing each error by the error obtained using some benchmark method of forecasting. Let $r_t = e_t/e_t^*$ denote the relative error where $e_t^*$ is the forecast error obtained from the benchmark method.  Usually the benchmark method is the naive method where $F_t$ is equal to the last observation. Then we can define 

Median Relative Absolute Error (MdRAE) = $median(|r_t |)$ 

Geometric Mean Relative Absolute Error (GMRAE) = $gmean(|r_t |)$

Because they are not scale dependent, these relative-error metrics were recommended in studies by Armstrong and Collopy (1992) and by Fildes (1992) for assessing forecast accuracy across multiple series. However, when the errors are small, as they can be with intermittent series, use of the naïve method as a benchmark is no longer possible because it would involve division by zero.

> 在计算与尺度无关的测量值时，除了百分数之外的另一种方法是将每个误差除以使用某种基准预测方法得到的误差。设$r_t = e_t/e_t^*$表示相对误差，其中$e_t^*$是通过benchmark方法得到的预测误差。通常，benchmark方法是一种简naive的方法，其中$F_t$等于最后一个观察值。然后我们可以定义
>
> Median Relative Absolute Error (MdRAE) = $median(|r_t |)$ 
>
> Geometric Mean Relative Absolute Error (GMRAE) = $gmean(|r_t |)$
>
> 由于它们 scale dependent，Armstrong和Collopy(1992)以及Fildes(1992)在研究中推荐使用这些相对误差度量来评估多个series的预测准确性。但是，当errors较小时（如 intermittent series 的误差一样），就不再可能使用naive方法作为benchmark，因为它将涉及除零。

## Scale-free errors

The MASE was proposed by Hyndman and Koehler (2006) as a generally applicable measurement of forecast accuracy without the problems seen in the other measurements. They proposed scaling the errors based on the *in-sample* MAE from the naïve forecast method. Using the naïve method, we generate one-period-ahead forecasts from each data point in the sample. Accordingly, a scaled error is defined as
$$
q_t = \frac{e_t}{\frac{1}{n-1}\sum_{i=2}^{n}|Y_i-Y_{i-1}|}
$$
The result is independent of the scale of the data. A scaled error is less than one if it arises from a better forecast than the average one-step, naïve forecast computed in- sample. Conversely, it is greater than one if the forecast is worse than the average one-step, naïve forecast computed in-sample.

> MASE由Hyndman和Koehler（2006）提出，是一种**普遍适用的对预测准确性的度量**，没有其他度量中出现的问题。他们从naive预测方法中提出了基于 in-sample 的MAE的误差缩放方法。使用naive的方法，我们从样本中的每个数据点生成一个提前期的预测。因此，定标误差定义为
> $$
> q_t = \frac{e_t}{\frac{1}{n-1}\sum_{i=2}^{n}|Y_i-Y_{i-1}|}
> $$
> 结果与数据规模无关。 如果缩放误差来自比平均单步naive预测的样本计算更好的预测，则该误差 < 1。 相反，如果预测比样本中计算的平均单步幼稚预测差，则 > 1。

The mean absolute scaled error is simply
$$
MASE = mean(|q_t|)
$$
The first row of Table 2 shows the intermittent series plotted

![Table 2](/Users/helloword/Anmingyu/Gor-rok/Metrics/Regression/MASE/table2.png)

in Figure 1. The second row gives the naive forecasts, which are equal to the previous actual values. The final row shows the naive-forecast errors. The denominator of $q_t$ is the mean of the shaded values in this row; that is the MAE of the naive method.

The only circumstance under which the MASE would be infinite or undefined is when all historical observations are equal.

> 平均绝对比例误差就是
> $$
> MASE = mean(|q_t|)
> $$
> Table2 的第一行显示了绘制的 intermittent series
>
> 在图1中。第二行给出了Naive的预测，该预测等于先前的实际值。 最后一行显示了naive的预测误差。 $ q_t $的分母是带阴影部分的值的平均值； 那就是naive的方法的MAE。
>
> MASE出现infinity或undefined唯一情况是所有历史观测值相等。

The in-sample MAE is used in the denominator because it is always available and it effectively scales the errors. In contrast, the out-of-sample MAE for the naïve method may be zero because it is usually based on fewer observations. For example, if we were forecasting only two steps ahead, then the out-of-sample MAE would be zero. If we wanted to compare forecast accuracy at one step ahead for ten different series, then we would have one error for each series. The out-of-sample MAE in this case is also zero. These types of problems are avoided by using in-sample, one-step MAE.

A closely related idea is the MAD/Mean ratio proposed by Hoover (this issue) which scales the errors by the in-sample mean of the series instead of the in-sample mean absolute error. This ratio also renders the errors scale free and is always finite unless all historical data happen to be zero. Hoover explains the use of the MAD/Mean ratio only in the case of in-sample, one-step forecasts (situation 2 of the three situations described in the introduction). However, it would also be straightforward to use the MAD/Mean ratio in the other two forecasting situations.

The main advantage of the MASE over the MAD/Mean ratio is that the MASE is more widely applicable. The MAD/Mean ratio assumes that the mean is stable over time (technically, that the series is “stationary”). This is not true for data which show trend, seasonality, or other patterns. While intermittent data is often quite stable, sometimes seasonality does occur, and this might make the MAD/Mean ratio unreliable. In contrast, the MASE is suitable even when the data exhibit a trend or a seasonal pattern.

The MASE can be used to compare forecast methods on a single series, and, because it is scale-free, to compare forecast accuracy across series. For example, you can average the MASE values of several series to obtain a measurement of forecast accuracy for the group of series. This measurement can then be compared with the MASE values of other groups of series to identify which series are the most difficult to forecast. Typical values for one-step MASE values are less than one, as it is usually possible to obtain forecasts more accurate than the naïve method. Multistep MASE values are often larger than one, as it becomes more difficult to forecast as the horizon increases.

The MASE is the only available accuracy measurement that can be used in all three forecasting situations described in the introduction, and for all forecast methods and all types of series. I suggest that it is the best accuracy metric for intermittent demand studies and beyond.

> 分母中使用了 in-sample MAE，因为它始终可用并且可以有效地缩放误差。 相比之下，朴素方法的样本外MAE可能为零，因为它通常基于较少的观察结果。 例如，如果我们仅预测了两个steps，则样本外MAE将为零。 如果我们想比较十个不同系列的预测准确性，那么每个系列都会有一个误差。 在这种情况下，样本外MAE也是零。 通过使用 in-sample 的 one-step MAE可以避免这些类型的问题。
>
> 一个密切相关的想法是 Hoover （此问题）提出的MAD / Mean ratio，该误差按 series 的 in-sample 平均值而不是 in-sample mae来缩放误差。这个比率还使误差没有规模限制，并且除非所有历史数据碰巧为零，否则误差始终是有限的。 Hoover仅在样本内一步预测（导言中描述的三种情况的情况2）的情况下解释了MAD /均值比率的使用。 但是，在其他两种预测情况下使用 MAD /mean ratio 也很简单。
>
> 与 MAD/Mean ratio 相比，MASE的主要优点是它的适用性更广。MAD/Mean ratio 假设 Mean 随时间的变化是稳定的(技术上讲，序列是“平稳的”)。对于显示trend、seasonality或其他 pattern 的数据，情况并非如此。虽然 intermitent数据通常相当稳定，但有时也会出现季节性，这可能会使 MAD/Mean ratio 变得不可靠。相比之下，即使数据显示出trend或seasonal pattern，MASE也适用。
>
> MASE可用于比较单个 series 的预测方法，并且由于它是scale-free的，因此可用于比较跨series的预测准确性。例如，您可以对多个series的MASE值求平均值，以获得对该series的预测准确度的度量。然后可以将该值与其他series组的MASE值进行比较，以确定哪个series是最难预测的。one-step MASE值的典型值  < 1，因为通常可以获得比naive方法更准确的预测。 multistep MASE值通常 > 1，因为随着范围的增加，它变得更加难以预测。
>
> MASE是在导言中介绍的所有三种预测情况下以及所有预测方法和所有类型series中都可以使用的 **唯一可用 度量**。 我建议，对于 intermittent-demand 研究及以后的研究，这是最佳的准确性指标。

## References

Armstrong, J. S. & Collopy F. (1992). Error measures for generalizing about forecasting methods: Empirical comparisons, *International Journal of Forecasting*, 8, 69–80.

Boylan, J. (2005). Intermittent and lumpy demand: A forecasting challenge, *Foresight: The International Journal of Applied Forecasting,* Issue 1, 36-42.

Fildes, R. (1992). The evaluation of extrapolative forecasting methods, *International Journal of Forecasting*, 8, 81–98.

Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy, *International Journal of Forecasting*. To appear.

Makridakis, S. & Hibon, M. (2000). The M3-competition: Results, conclusions and implications, *International Journal of Forecasting*, 16, 451–476.

Makridakis, S. G., Wheelwright, S. C. & Hyndman, R. J. (1998). *Forecasting: Methods and Applications* (3rd ed.), New York: John Wiley & Sons.

Syntetos, A. A. & Boylan, J. E, (2005). The accuracy of intermittent demand estimates, *International Journal of Forecasting*, 21, 303-314.

> Contact Info:
>  Rob J. Hyndman
>  Monash University, Australia Rob.Hyndman@buseco.monash.edu