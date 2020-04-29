# Top 平台电商 GMV 的预测与分析

刚刚看到的新闻 “天猫双 11 第一小时收官战报：571 亿 GMV”。哇，又刷新纪录了。

什么是GMV？为什么平台电商 （e commerce）都非常关注这个指标。我今天就来聊聊GMV的预测与分析。

投行出来后，有过一段时间不成功的创业经历后，去了一家 top 平台类电商做 FP&A 的consultant，专注于 GMV 的预测与分析。

GMV，Gross Merchandise Volume，也就是一段时间内的成交总额。对于平台类电商来说，这是最重要的一个指标，因为平台收取的交易费都是基于GMV的。交易费又会转化成公司的income，再由此推算出相应的费用。因此，对于GMV的预测与分析就尤为重要。而电商这个行业又特别的 dynamic，同业的竞争，政府的监管，消费者行为与习惯的改变，节假日的到来，都对发生在这个平台上的交易量有着显著的影响。所以，我们需要构建一套稳健而又灵活的 GMV 预测分析方法与系统。

每年的8月~9月，也就在是Q3，我们会做 annual budget and forecast，对下一财年的 GMV 做预测。一个简化的分析预测框架大致如下：

- GMV Weekly Index Baseline

- Seasonality adjustment

- - Chinese New Year
  - Easter Day
  - Labor Day
  - Independence Day
  - Halloween
  - Thanksgiving & Black Friday
  - Christmas and Year End

- Event Adjustment

- Site/Corridor Demand and Share assumptions

下面我用一个 sample 来具体讲讲 GMV Weekly Index Baseline。

我假设这个电商平台在2014/2015/2016年第一周的 GMV 分别是 60/80/100，然后用一个 excel 的随机函数（for example, B5=$B$4+RANDBETWEEN(-50,50)/3）模拟后面每周的 GMV 。再计算一个 Weekly Index Base (for example, C1=AVERAGE($B$14:$B$18) )，最后算出一个 Weekly Index Number (for example, C4=B4/$C$1*100 )。画出的 chart 如下。

![](/Users/helloword/Anmingyu/Gor-rok/GMVPrediction/Top平台电商GMV的预测与分析/1.png)

![](/Users/helloword/Anmingyu/Gor-rok/GMVPrediction/Top平台电商GMV的预测与分析/2.png)

![](/Users/helloword/Anmingyu/Gor-rok/GMVPrediction/Top平台电商GMV的预测与分析/3.png)

仔细观察上面两张 charts，直观的看，有什么区别呢？

第一张图 （Weekly GMV）相对于第二张图 （Weekly GMV Index）更加的 disperse。Weekly GMV Index 这张图可以看出是围绕着 100 这个值上下波动。为什么会这样呢？原因很简单，请自行思考，^_^ 其实，真实的 GMV 图形不会像这个样子，要考虑的因素更多，计算也会更复杂，这里只是 for demonstration purpose。

有了 historical GMV Index pattern，我们就可画出未来的 GMV Index 的图形，再倒算出 GMV 的值。在预测时，还要做相应的 seasonality/event 的 adjustments。

至于 Seasonality adjustment，是因为在一些特殊的日子里（通常是一些节日），GMV 图形的 pattern 会比较特殊，我们在预测时会做相应的调整。

至于 Event adjustment，比如去年5月时，我们为了促销做了一些活动，GMV 当然在那段时间就会表现的特别的好，但是，我们无法保证在明年5月也会做这样的促销。所以，在 forecast GMV 时，要去除这个 event 对于 GMV 的影响。

至于 Site/Corridor Demand and Share assumptions，因为我们的 GMV 来自于不同的国家和地区，其 pattern 也是不一样的。要做更好的预测，我们还要把 GMV 分的更加细致一些去看。

好了，今天给大家简单讲了一个平台类电商 GMV 分析预测的思路和框架，希望对你有用。