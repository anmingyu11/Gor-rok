## 4 EXPERIMENTS

在本节中，我们首先介绍训练 listing embedding 及其离线评估的细节。然后，我们在 listing 页面上展示了使用 listing embedding 的类似 listing 推荐的线上结果。最后，给出了搜索排序模型的背景，并描述了如何利用 listing embeding 和 listing type embedding 和 user type embedding 实现搜索中的实时个性化特征。两种 embedding 应用都成功地投入生产。

#### 4.1 Training Listing Embeddings

对于训练 listing embedding，我们通过从已登录用户获取所有搜索记录，然后根据用户ID对它们进行分组并按时间排序 listing id 来创建 8 亿个点击 session 。 然后，根据 30分钟的固定规则将一个较大的 listing ID 的有序列表分成多个。 接下来，我们移除了意外点击和短暂点击，即用户在 listing 页面停留不到 30秒的点击，仅保留了包含 2 次或更多点击的 session。 最后，通过删除用户 ID 列来将 session 匿名化。 如前所述，点击 session 由探索 session＆预订 session（以预订结束的点击顺序）组成。 根据离线评估结果，我们在训练数据中对预订的 session 进行了 5 倍的过采样，从而获得了效果最好的 listing embedding。

#### Setting up Daily Training.

我们学习了450万个 Airbnb listings 的 listing embedding，并且我们的训练数据实用性和参数已使用下面介绍的离线评估技术进行了调整。通过处理最新的日期搜索 session 并将其添加到数据集，并从数据集中丢弃最早的日期搜索 session，我们的训练数据会在几个月内以滑动窗口的方式每天更新。我们为每个 $listing\_{id}$ 训练 embedding，在训练之前我们在其中随机初始化向量（每次都使用相同的随机种子）。我们发现，如果我们每天从头开始重新训练 listing embedding，而不是对现有向量进行渐进式训练，则会获得更好的离线性能。每天的向量差异不会导致我们模型中的差异，因为在我们的应用中，我们使用余弦相似性作为主要信号，而不是实际向量本身。即使向量随时间变化，余弦相似性度量的内涵及其范围也不变。

listing embedding 的维数设置为 $d = 32$，因为我们发现要在离线性能和为实时相似度计算目的将向量存储在搜索机器的 RAM 内存中所需的内存之间进行权衡取舍。 上下文窗口大小设置为 $m = 5$，我们对训练数据执行了 $10$ 次迭代。 为了进行聚集搜索，我们修改了原始 word2vec code1。 训练使用了 MapReduce，其中 300个 mappers 读取数据，而单个 reducer 以多线程方式训练模型。 End-to-end 数据生成和培训 pipeline 是使用 Airflow2 实现的，Airflow2 是 Airbnb 的开源调度平台。

#### 4.2 Offline Evaluation of Listing Embeddings

为了能够对优化函数、训练数据、超参数等方面的调整做出快速决策，我们需要一种快速比较不同 embedding 方式的方法。

**评估训练好的 embedding 的一种方法是，基于用户最近的点击，测试它们在推荐用户可能预订的 listing 有多好。具体地说，让我们假设给了我们最近点击的 listing 和需要排序的候选 listing ，其中包含用户最终预订的 listing 。通过计算点击 listing 和候选 listing 的 embedding 之间的余弦相似度，我们可以对候选 listing 进行排序，并观察预订 listing 的排序位置。**

出于评估的目的，我们使用了大量这样的 搜索、点击和预订活动，这些活动的排名已经由我们的搜索排序模型分配了。在图6中，我们显示了离线评估的结果，其中我们比较了 $d=32$ 的 Embedding 的几个版本，以了解它们如何根据之前的点击对预订的 listing 进行排名。对导致预订的每次点击的预订 listing 排名进行平均，最早可追溯到预订前的 17 次点击，也可以追溯到预订前的最后一次点击。值越低表示级别越高。我们比较的 embedding 版本是

1. d32: 用 (3) 训练
2. d32 book: 将预订作为全局 context 训练 (4) 
3. d32 book + neg: 将预订作为全局 context 训练并且从同一市场获取负样本(5).

可以看到，搜索排序模型随着点击次数的增加而变得更好，因为它使用了记忆功能。还可以观察到，基于 embedding 相似度对 listing 进行重新排序将是有用的，特别是在搜索漏斗的早期阶段。

相同类型的图表被用来做出关于超参数、数据构造等的决策。

![Figure6](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig6.png)

**Figure 6: Offline evaluation of Listing Embeddings**

#### 4.3 Similar Listings using Embeddings

每个 Airbnb 房屋 listing 页面 [3](https://zh.airbnb.com/rooms/433392?_set_bev_on_new_domain=1613826695_OGU2OTNlMzE0ODRh&translate_ugc=false&source_impression_id=p3_1613826703_2SMVzxdH2AbWTwNO) 都包含“相似的 listing”轮播，该列表会推荐与之相似且适用于相同日期的 listing。

在我们进行测试时，用于类似 listing 轮播的现有算法是在与给定 listing 相同的位置调用 main Search Ranking 模型，然后通过 可用性，价格范围和 listing 类型 过滤指定的 listing 。

我们进行了A/B测试，将现有的 listing 相似度算法与基于 embedding 的解决方案进行了比较，在基于 embedding 的解决方案中，通过在 listing embedding 空间中寻找 k 近邻来寻找相似的 listing。已知的学习到的 listing embedding ，通过计算来自同一市场的同一组日期(如果设置了签入和签出日期)的所有 listing 的向量 $\textbf{v}_l$ 和向量 $\textbf{v}_j$ 之间的余弦相似性，可以找到给定 listing $l$ 的相似 listing。取相似度最高的 k 个 listing 。计算是在线使用我们的 sharded 架构并行进行，其中 embedding 的部分存储在每个搜索机上。

A/B测试显示，基于 embedding 的解决方案使相似 listing 轮播的 CTR 增加了 21%(在 listing 页面输入日期的情况下增加 23%，在无日期页面的情况下增加 20%)，并使找到该 listing 的 guest 增加 4.9%(最终预订)。根据这些结果，我们将基于 embedding 的 相似 listing 部署到生产中。

#### 4.4 Real time personalization in Search Ranking using Embeddings

#### Background.

为了形式化描述我们的搜索排序模型，假设我们获得了有关每个搜索的训练数据 $D_s =(\textbf{x}_i，y_i),\ i = 1,\cdots,K$，其中 $K$ 是搜索返回的 listing 数量，$\textbf{x}_i$ 是向量包含第 $i$ 个 listing 的特征，且 $y_i \in \{0,0.01,0.25,1,-0.4\}$ 是分配给第 $i$ 个 listing 结果的 label 。要将 label 分配给搜索结果中的特定 listing，我们会在搜索发生后等待 1 周以观察最终结果，如果预订了 listing，$y_i = 1$，如果 guest 联系了 host 但没有预订，$y_i = 0.25$，如果 listing host 拒绝了  guest，$y = −0.4$ (注：此处应该是 typo,应为 $y_i$)，点击 listing 时 $y_i = 0.01$，浏览了 listing 但未点击时 $y_i = 0$。

经过 1 周的等待之后，集合 $D_s$ 也被缩短，只保留到用户最后一次点击 $K_c \le K$ 的结果。(注：这里没明白是什么意思)最后，为了生成数据 $\mathcal{D} = \bigcup_{s=1}^{N} D_s$，我们仅保留至少包含一个预订 label 的$D_s$ 。 每次我们训练新的 ranking model时，我们都会使用最近 30 天的数据。

第 $i$ 个 listing 的特征向量 $\textbf{x}_i$ 由 listing 特征, user 特征, query 特征和 cross 特征组成.

- listing feature：是与 listing 本身相关的 feature，例如每晚价格，listing 类型，房间数量，拒绝率等。
- query feature：是与查询相关的特征，如访客数量、停留时间、提前天数等。
- user feature：是与正在进行搜索的用户相关的特征，如平均预订价格、guest 评级等。
- cross feature：是从 listing, user, query  特征中衍生的特征。

例如：

- query listing distance：query 位置与 listing 位置之间的距离。
- capacity fit：客人查询数量与 listing 容纳人数之间的差异。
- price difference：listing 价格与用户历史预订的平均价格之间的差异。
- rejection probability：host 拒绝这些 query 参数的概率。
- click percentage：实时记忆 feature，跟踪用户点击该特定 listing 的百分比等。

该模型使用了大约 100 个 feature。为了简洁起见，我们就不一一列举了。

接下来，我们将问题表述为 pairwise 回归，并使用搜索 label 作为实用工具，并使用数据 $\mathcal{D}$ 来训练 GBDT 模型，并使用经过修改以支持 Lambda Rank 的软件包 4。 在离线评估不同模型时，我们在保留的搜索 session 集上使用 NDCG（标准的排序指标），即 $\mathcal{D}$ 的 80% 用于训练，而 20% 用于测试。

最终，模型经过训练后，即可用于搜索 listing 的在线评分。 用户 $u$ 执行的搜索查询 $q$ 返回的每个 listing 的特征向量  $\textbf{x}_i$ 都是线上计算的，并且使用分片架构并行进行评分。 在给定所有的分数后，按降序顺序向用户展示 listing 。

最后，一旦模型被训练，它被用于搜索 listing 的在线评分。对于用户 $u$ 执行的搜索查询 $q$ 返回的每个 listing，计算特征向量 $\textbf{x}_i$ 都是线上计算的，并且使用我们的分片架构并行评分 4。给定所有的分数后，将按照降序顺序向用户显示 listing。

#### Listing Embedding Features. 

在我们的搜索排序模型中添加 embedding feature 的第一步是将 450万个 embedding feature 加载到我们的搜索后端，这样它们就可以实时访问，进行特征计算和模型评分。

接下来，我们介绍了几个用户历史 short-term 集合，它们保存了最近两周的用户行为，当新的用户行为发生时，它们会实时更新。该逻辑是使用 Kafka 5 实现的。具体来说，对于每个 user_id，我们收集并维护 (定期更新) 以下一组 listing id:

1. $H_c$：**clicked listing_ids** -用户在过去2周点击的 listings。
2. $H_{l_c}$：**long-clicked listing_ids** -用户点击并在 listing 页面上停留超过 60 秒 的 listing。
3. $H_s$：**skipped listing_ids** -用户跳过而选择点击排名较低的 listing 的 listing。
4. $H_w$：**wishlists listing_ids** - 用户过去两周内添加到 wishlist 的 listings。
5. $H_i$：**queries listing_ids**——用户在过去两周内联系但没有预订的 listing 。
6. $H_b$ : **booked listing_ids** -用户在过去两周预订的 listing。

我们进一步将每个 short-term 集合 $H_∗$ 分成同一市场的子集。例如，如果用户点击了来自纽约和洛杉矶的 listing，它们的 $H_c$ 将进一步分为 $H_c(NY)$ 和$H_c(LA)$。

最后，我们定义了 embedding 特征，这些特征利用定义的好的集合 和 listing embedding 为每个候选 listing 生成分数。 表6 中汇总了这些 feature。

下面我们将描述 $EmbClickSim$ 特征是如何使用 $H_c$ 计算的。使用相应的用户短期历史记录集 $H_*$，以相同的方式计算 表6 顶部各行中的其余 feature。

为了计算候选 listing $l_i$ 的 $EmbClickSim$，我们需要计算其 listing embedding $v_{l_i}$与 $H_c$ 中 listing embedding 之间的余弦相似度。为此，我们首先计算 $H_c$ 在市场级别上的 centroid embedding。为了说明问题，我们假设 $H_c$ 包含来自 NY 的 5 个 listing 和来自 LA 的 3 个 listing。这将需要通过计算两个市场级别的 centroid embedding，一个用于 NY(纽约)，一个用于 LA(洛杉矶)。

最后，计算 $EmbClickSim$ 为 listing embedding $\textbf{v}_{l_i}$ 与 $H_c$ 市场级别的 centroid embedding 的两个点中的最相似的。

$EmbClickSim$  表示为：
$$
EmbClickSim(l_i,H_c) = \mathop{max}_{m \in M} 
\
cos(\textbf{v}_{l_i},\sum_{l_h \in m, l_h \in H_c}\textbf{v}_{l_h})

\qquad (10)
$$
其中 $M$ 是用户曾经点击过的市场集合

除了与所有用户点击的相似性外，我们还添加了一个 feature，用于测量与latest long click 的相似性，即 $EmbLastLongClickSim$。对于候选 listing $l_i$，通过找出其 embedding $\textbf{v}_{l_i}$ 与来自 $H_{l_c}$ 的 latest long click $l_{last}$ 的 embedding 之间的余弦相似性来计算，
$$
EmbLastLongClickSim(l_i,H_{l_c})=cos(\textbf{v}_{l_i},\textbf{v}_{l_{last}})
\qquad (11)
$$
![Table6](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table6.png)

**Table 6: Embedding Features for Search Ranking**

#### User-type & Listing-type Embedding Features.

我们遵循类似的过程来引入基于 user type 和 listing type 的 embedding feature。我们使用 5000万 用户预订 session 训练了 500K user type 和 500K  listing type 的 embedding。embedding 是 $d=32$ 维的，并且在预订过程中使用 $m=5$ 的滑动窗口进行训练。user type 和 listing type 的 embedding 被加载到搜索机内存中，以便我们可以在线计算类型相似度。

为了计算候选 listing $l_i$ 的 $UserTypeListingTypeSim$ 特征，我们简单地查找其当前 listing type $l_t$ 以及执行搜索的用户的当前 user type $u_t$ 并计算其 embedding 之间的余弦相似性，
$$
UserTypeListingTypeSim(u_t,l_t) = cos(\textbf{v}_{u_t},\textbf{v}_{l_t})
\qquad (12)
$$
表6 中的所有 feature 都被记录 $30$ 天，这样它们就可以添加到搜索排序的训练集 $\mathcal{D}$ 中。表7 报告了 feature 的覆盖范围，即具有特定 feature 的 $\mathcal{D}$ 的比例。正如预期的那样，可以观察到基于用户 click 和 skip 的 feature 覆盖率最高。

最后，通过添加 embedding feature 训练出新的 GBDT 搜索排序模型。embedding 特征的特征重要性(104个特征中的排名)如 表7 所示。

排名最高的 feature 是 listings user clicked的相似度($EmbClickSim$：总体排名第5)，和 listings user skiped 的相似度（$ EmbSkipSim $：总体排名第8）。 前 20 个 feature 中有 5 个 embedding features。不出所料，长期 feature $UserTypeListingTypeSim$ 使用了所有过去的用户预订，其排名比短期 feature $ EmbBookSim $更好，后者仅考虑了最近两周的预订。这还表明基于过去预订的推荐能够更好地使用使用历史预订 session 训练的 embedding 而不是点击 session 训练的 embedding(注：这里是从特征重要性角度来讲的)。

为了评估模型是否学会了按照我们的预期使用这些特征，我们绘制了 3 个 embedding fatures 的 PDP : $EmbClickSim$、$EmbSkipSim$ 和 $UserTypeListTypeSim$。

这些图显示了如果我们固定除了我们正要查看的 feature 以外的所有 feature 的值，将会对 listing 的排序得分产生什么影响。

在左边的子图中，我们可以看到 $EmbClickSim$ 的值(表示 listing 与用户最近点击的 listing 相似度)越大模型得分越高。

在中间的子图中，$EmbSkipSim$ 的值(表示 listing 与用户跳过的 listing 的相似度)越大，模型得分越低。

右边的子图中，$UserTypeListingTypeSim$ 的值(user 类型与 listing 类型的相似度)越高，模型得分越高。

![Table7](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table7.png)

**Table 7: Embedding Features Coverage and Importances**

![Figure7](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig7.png)

**Figure 7: Partial Dependency Plots for EmbClickSim, EmbSkipSim and UserTypeListTypeSim**

#### Online Experiment Results Summary.

我们进行了离线和在线实验(A / B测试)。 首先，我们比较了在具有和不具有 embedding feature 的情况下针对相同数据训练的两个搜索排序模型。 在表 8中，我们根据每个应用(impression(应该指的是曝光)，click，rejection和 booking) 的 DCU (Discounted Cumulative Utility) 和整体 NDCU（Normalized Discounted Cumulative Utility）汇总了结果。 

可以看出，加入 embedding feature使 NDCU 提升了 2.27%，booking DCU 增长了 2.58%，这意味着 booking listing 在 hold-out 集合中排名更高，没有任何 rejection (DCU -0.4 是持平的) ，这意味着被拒绝的 listing 不会比没有 embedding 特征的模型中的排名更高。

从表8中的观察结果，再加上 embedding feature 在 GBDT 特征重要性排名靠前的事实(表7)，以及发现 feature 的作用符合我们直观的预期(图7)，足以决定继续进行线上实验。线上实验中，我们看到了在 booking 上的明显收益，embedding feature 也开始投入生产中。几个月后，我们进行了一次反向测试，删除 embedding feature，结果导致了 booking 的负面影响，这是实时 embedding feature 有效的另一个指标。

![Table8](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table8.png)

**Table 8: Offline Experiment Results**

## 5 CONCLUSION

我们提出了一种在Airbnb搜索排序中进行实时个性化的新方法。该方法基于用户 click 和 bokking session 中的上下文共现来学习主页 listing 和用户的低维表示。为了更好地利用可用的搜索 context，我们将全局 context 和显式 negative signal 等概念纳入训练流程。我们在相似的 listing 推荐和搜索排序中对该方法进行了评估。在对实时搜索流量进行成功测试后，两个 embedding 应用程序都被部署到生产环境中。



-----------------------

## QA

1. Setting up Daily Training.

   > 4

2. Embedding 的离线评估

   > 4