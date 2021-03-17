## 4 EXPERIMENTS

In this section we first cover the details of training Listing Embeddings and their Offline Evaluation. We then show Online Experiment Results of using Listing Embeddings for Similar Listing Recommendations on the Listing Page. Finally, we give background on our Search Ranking Model and describe how Listing Embeddings and Listing Type & User Type Embeddings were used to implement features for Real-time Personalization in Search. Both applications of embeddings were successfully launched to production.

> 在本节中，我们首先介绍训练 listing embedding 及其离线评估的细节。然后，我们在 listing 页面上展示了使用 listing embedding 的类似 listing 推荐的线上结果。最后，给出了搜索排序模型的背景，并描述了如何利用 listing embeding 和 listing type embedding 和 user type embedding 实现搜索中的实时个性化特征。两种 embedding 应用都成功地投入生产。

#### 4.1 Training Listing Embeddings

For training listing embeddings we created 800 million click sessions from search, by taking all searches from logged-in users, grouping them by user id and ordering clicks on listing ids in time. This was followed by splitting one large ordered list of listing ids into multiple ones based on 30 minute inactivity rule. Next, we removed accidental and short clicks, i.e. clicks for which user stayed on the listing page for less than 30 seconds, and kept only sessions consisting of 2 or more clicks. Finally, the sessions were anonymized by dropping the user id column. As mentioned before, click sessions consist of exploratory sessions &. booked sessions (sequence of clicks that end with booking). In light of offline evaluation results we oversampled booked sessions by 5x in the training data, which resulted in the best performing listing embeddings.

#### Setting up Daily Training.

We learn listing embeddings for 4.5 million Airbnb listings and our training data practicalities and parameters were tuned using offline evaluation techniques presented below. Our training data is updated daily in a sliding window manner over multiple months, by processing the latest day search sessions and adding them to the dataset and discarding the oldest day search sessions from the dataset. We train embeddings for each listing\_id, where we initialize vectors randomly before training (same random seed is used every time). We found that we get better offline performance if we re-train listing embeddings from scratch every day, instead of incrementally continuing training on existing vectors. The day-to-day vector differences do not cause discrepancies in our models because in our applications we use the cosine similarity as the primary signal and not the actual vectors themselves. Even with vector changes over time, the connotations of cosine similarity measure and its ranges do not change.

Dimensionality of listing embeddings was set to $d = 32$, as we found that to be a good trade-off between offline performance and memory needed to store vectors in RAM memory of search machines for purposes of real-time similarity calculations. Context window size was set to $m = 5$, and we performed $10$ iterations over the training data. To implement the congregated search change to the algorithm we modified the original word2vec c code1 . Training used MapReduce, where 300 mappers read data and a single reducer trains the model in a multi-threaded manner. End-to-end daily data generation and training pipeline is implemented using Airflow2 , which is Airbnb’s open-sourced scheduling platform.

> 对于训练 listing embedding，我们通过从已登录用户获取所有搜索记录，然后根据用户ID对它们进行分组并按时间排序 listing id来创建 8 亿个点击 session 。 然后，根据 30分钟的固定规则将一个较大的 listing ID的有序列表分成多个。 接下来，我们移除了意外点击和短暂点击，即用户在 listing 页面停留不到 30秒的点击，仅保留了包含 2 次或更多点击的 session。 最后，通过删除用户 ID 列来将 session 匿名化。 如前所述，点击 session 由探索 session＆预订 session（以预订结束的点击顺序）组成。 根据离线评估结果，我们在训练数据中对预订的 session 进行了 5 倍的过采样，从而获得了效果最好的 listing embedding。
>
> 我们学习了450万个 Airbnb listings的 listing embedding，并且我们的训练数据实用性和参数已使用下面介绍的离线评估技术进行了调整。通过处理最新的日期搜索 session 并将其添加到数据集，并从数据集中丢弃最早的日期搜索 session，我们的训练数据会在几个月内以滑动窗口的方式每天更新。我们为每个 $listing\ _id$ 训练 embedding，在训练之前我们在其中随机初始化向量（每次都使用相同的随机种子）。我们发现，如果我们每天从头开始重新训练 listing embedding，而不是对现有向量进行渐进式训练，则会获得更好的离线性能。每天的向量差异不会导致我们模型中的差异，因为在我们的应用中，我们使用余弦相似性作为主要信号，而不是实际向量本身。即使向量随时间变化，余弦相似性度量的内涵及其范围也不变。
>
> listing embedding 的维数设置为 $d = 32$，因为我们发现要在离线性能和为实时相似度计算目的将向量存储在搜索机器的 RAM 内存中所需的内存之间进行权衡取舍。 上下文窗口大小设置为 $m = 5$，我们对训练数据执行了 $10$ 次迭代。 为了进行聚集搜索，我们修改了原始 word2vec c code1。 训练使用了 MapReduce，其中 300个 mappers 读取数据，而单个 reducer 以多线程方式训练模型。 End-to-end 数据生成和培训 pipeline 是使用Airflow2实现的，Airflow2 是 Airbnb 的开源调度平台。

#### 4.2 Offline Evaluation of Listing Embeddings

To be able to make quick decisions regarding different ideas on optimization function, training data construction, hyperparameters, etc, we needed a way to quickly compare different embeddings.

One way to evaluate trained embeddings is to test how good they are in recommending listings that user would book, based on the most recent user click. More specifically, let us assume we are given the most recently clicked listing and listing candidates that need to be ranked, which contain the listing that user eventually booked. By calculating cosine similarities between embeddings of clicked listing and candidate listings we can rank the candidates and observe the rank position of the booked listing.

For purposes of evaluation we use a large number of such search, click and booking events, where rankings were already assigned by our Search Ranking model. In Figure 6 we show results of offline evaluation in which we compared several versions of $d = 32$ embeddings with regards to how they rank the booked listing based on clicks that precede it. Rankings of booked listing are averaged for each click leading to the booking, going as far back as 17 clicks before the booking to the Last click before the booking. Lower values mean higher ranking. Embedding versions that we compared were 

1. d32: trained using (3)
2. d32 book: trained with bookings as global context (4) 
3. d32 book + neg: trained with bookings as global context and explicit negatives from same market (5).

It can be observed that Search Ranking model gets better with more clicks as it uses memorization features. It can also be observed that re-ranking listings based on embedding similarity would be useful, especially in early stages of the search funnel. Finally, we can conclude that d32 book + neg outperforms the other two embedding versions. The same type of graphs were used to make decisions regarding hyperparameters, data construction, etc.

> 为了能够对优化函数、训练数据、超参数等方面的调整做出快速决策，我们需要一种快速比较不同embedding方式的方法。
>
> 评估训练好的 embedding 的一种方法是，基于用户最近的点击，测试它们在推荐用户可能预订的 listing 有多好。具体地说，让我们假设给了我们最近点击的 listing 和需要排序的候选 listing ，其中包含用户最终预订的 listing 。通过计算点击 listing 和候选 listing 的 embedding 之间的余弦相似度，我们可以对候选 listing 进行排序，并观察预订 listing 的排序位置。
>
> 出于评估的目的，我们使用了大量这样的 搜索、点击和预订活动，这些活动的排名已经由我们的搜索排序模型分配了。在图6中，我们显示了离线评估的结果，其中我们比较了 $d=32$ 的Embedding的几个版本，以了解它们如何根据之前的点击对预订的 listing 进行排名。对导致预订的每次点击的预订 listing 排名进行平均，最早可追溯到预订前的17次点击，也可以追溯到预订前的最后一次点击。值越低表示级别越高。我们比较的 embedding 版本是
>
> 1. d32: 用 (3) 训练
> 2. d32 book: 将预订作为全局 context 训练 (4) 
> 3. d32 book + neg: 将预订作为全局 context 训练并且从同一市场获取负样本(5).
>
> 可以看到，搜索排序模型随着点击次数的增加而变得更好，因为它使用了记忆功能。还可以观察到，基于 embedding 相似度对 listing 进行重新排序将是有用的，特别是在搜索漏斗的早期阶段。
>
> 最后，我们可以得出结论，d32 book + neg 的表现优于其他两个 embedding 版本。
> 相同类型的图表被用来做出关于超参数、数据构造等的决策。

![Figure6](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig6.png)

**Figure 6: Offline evaluation of Listing Embeddings**

#### 4.3 Similar Listings using Embeddings

Every Airbnb home listing page3 contains Similar Listings carousel which recommends listings that are similar to it and available for the same set of dates. At the time of our test, the existing algorithm for Similar Listings carousel was calling the main Search Ranking model for the same location as the given listing followed by filtering on availability, price range and listing type of the given listing.

We conducted an A/B test where we compared the existing similar listings algorithm to an embedding-based solution, in which similar listings were produced by finding the k-nearest neighbors in listing embedding space. Given learned listing embeddings, similar listings for a given listing l were found by calculating cosine similarity between its vector $\textbf{v}_l$ and vectors $\textbf{v}_j$ of all listings from the same market that are available for the same set of dates (if check-in and check-out dates are set). The $K$ listings with the highest similarity were retrieved as similar listings. The calculations were performed online and happen in parallel using our sharded architecture, where parts of embeddings are stored on each of the search machines.

The A/B test showed that embedding-based solution lead to a 21% increase in Similar Listing carousel CTR (23% in cases when listing page had entered dates and 20% in cases of dateless pages) and 4.9% increase in guests who find the listing they end up booking in Similar Listing carousel. In light of these results we deployed the embedding-based Similar Listings to production.

> 每个 Airbnb 房屋 listing 页面 [3](https://zh.airbnb.com/rooms/433392?_set_bev_on_new_domain=1613826695_OGU2OTNlMzE0ODRh&translate_ugc=false&source_impression_id=p3_1613826703_2SMVzxdH2AbWTwNO) 都包含“相似的 listing”轮播，该列表会推荐与之相似且适用于相同日期的 listing。
>
> 在我们进行测试时，用于类似 listing 轮播的现有算法是在与给定 listing 相同的位置调用 main Search Ranking 模型，然后通过 可用性，价格范围和 listing类型 过滤指定的 listing 。
>
> 我们进行了A/B测试，将现有的 listing 相似度算法与基于 embedding 的解决方案进行了比较，在基于 embedding 的解决方案中，通过在 listing embedding 空间中寻找 k 近邻来寻找相似的 listing。已知的学习到的 listing embedding ，通过计算来自同一市场的同一组日期(如果设置了签入和签出日期)的所有 listing 的向量 $\textbf{v}_l$和向量 $\textbf{v}_j$ 之间的余弦相似性，可以找到给定 listing $l$ 的相似 listing。取相似度最高的 K 个 listing 。计算是在线使用我们的 sharded 架构并行进行，其中 embedding 的部分存储在每个搜索机上。
>
> A/B测试显示，基于 embedding 的解决方案使相似 listing 轮播的 CTR 增加了 21%(在 listing 页面输入日期的情况下增加 23%，在无日期页面的情况下增加 20%)，并使找到该 listing 的 guest 增加 4.9%(最终预订)。根据这些结果，我们将基于 embedding 的 相似 listign 部署到生产中。

#### 4.4 Real time personalization in Search Ranking using Embeddings

#### Background.

To formally describe our Search Ranking Model, let us assume we are given training data about each search $D_s =(\textbf{x}_i，y_i),\ i = 1,\cdots,K$ , where $K$ is the number of listings returned by search, $\textbf{x}_i$ is a vector containing features of the i-th listing result and $y_i \in \{0,0.01,0.25,1,-0.4\}$ is the label assigned to the $i$-th listing result. To assign the label to a particular listing from the search result we wait for $1$ week after search happened to observe the final outcome, which can be $y_i = 1$ if listing was booked, $y_i = 0.25$ if listing host was contacted by the guest but booking did not happen, $y = −0.4$ if listing host rejected the guest, $y_i = 0.01$ is listing was clicked and $y_i = 0$ if listing was just viewed but not clicked. After that $1$ week wait the set $D_s$ is also shortened to keep only search results up to the last result user clicked on $K_c \le K$. Finally, to form data $\mathcal{D} = \bigcup_{s=1}^{N} D_s$ we only keep $D_s$ sets which contain at least one booking label. Every time we train a new ranking model we use the most recent 30 days of data.

Feature vector $\textbf{x}_i$ for the $i$-th listing result consists of listing features, user features, query features and cross-features.

- Listing features are features associated with the listing itself, such as price per night, listing type, number of rooms, rejection rate, etc. 
- Query features are features associated with the issued query, such as number of guests, length of stay, lead days, etc. 
- User features are features associated with the user who is conducting the search, such as average booked price, guest rating, etc. 
- Cross-features are features derived from two or more of these feature sources: listing, user, query. 

Examples of such features are 

- query listing distance: distance between query location and listing location
- capacity fit: difference between query number of guests and listing capacity
- price difference: difference between listing price and average price of user’s historical bookings
- rejection probability: probability that host will reject these query parameters
- click percentage: real-time memorization feature that tracks what percentage of user’s clicks were on that particular listing, etc. 

The model uses approximately 100 features. For conciseness we will not list all of them.

Next, we formulate the problem as pairwise regression with search labels as utilities and use data $\mathcal{D}$ to train a Gradient Boosting Decision Trees (GBDT) model, using package 4 that was modified to support Lambda Rank. When evaluating different models offline, we use NDCG, a standard ranking metric, on hold-out set of search sessions, i.e. 80% of $\mathcal{D}$ for training and 20% for testing.

Finally, once the model is trained it is used for online scoring of listings in search. The signals needed to calculate feature vectors $\textbf{x}_i$ for each listing returned by search query $q$ performed by user $u$ are all calculated in an online manner and scoring happens in parallel using our sharded architecture. Given all the scores, the listings are shown to the user in a descending order of predicted utility.

> 为了形式化描述我们的搜索排序模型，假设我们获得了有关每个搜索的训练数据 $D_s =(\textbf{x}_i，y_i),\ i = 1,\cdots,K$，其中 $K$ 是搜索返回的 listing 数量，$\textbf{x}_i$ 是向量包含第 $i$ 个 listing 的特征，且 $y_i \in \{0,0.01,0.25,1,-0.4\}$ 是分配给第 $i$ 个 listing 结果的 label 。要将 label 分配给搜索结果中的特定 listing，我们会在搜索发生后等待 1 周以观察最终结果，如果预订了 listing，$y_i = 1$，如果 guest 联系了 host 但没有预订，$y_i = 0.25$，如果 listing host 拒绝了  guest，$y = −0.4$ (注：此处应该是typo,应为 $y_i$)，点击 listing 时 $y_i = 0.01$，浏览了 listing 但未点击时 $y_i = 0$。
>
> 经过 1 周的等待之后，集合 $D_s$ 也被缩短，只保留到用户最后一次点击 $K_c \le K$ 的结果。(注：这里没明白是什么意思)最后，为了生成数据 $\mathcal{D} = \bigcup_{s=1}^{N} D_s$，我们仅保留至少包含一个预订 label 的$D_s$ 。 每次我们训练新的 ranking model时，我们都会使用最近 30 天的数据。
>
> 第 $i$ 个 listing 的特征向量 $\textbf{x}_i$ 由 listing 特征, user 特征, query 特征 和 cross特征组成.
>
> - listing feature：是与 listing 本身相关的 feature，例如每晚价格，listing 类型，房间数量，拒绝率等。
> - query feature：是与查询相关的特征，如访客数量、停留时间、提前天数等。
> - user feature：是与正在进行搜索的用户相关的特征，如平均预订价格、guest 评级等。
> - cross feature：是从 listing, user, query  特征中衍生的特征。
>
> 例如：
>
> - query listing distance：query 位置与 listing 位置之间的距离。
> - capacity fit：客人查询数量与 listing 容纳人数之间的差异。
> - price difference：listing 价格与用户历史预订的平均价格之间的差异。
> - rejection probability：host 拒绝这些 query 参数的概率。
> - click percentage：实时记忆 feature，跟踪用户点击该特定 listing 的百分比等。
>
> 该模型使用了大约 100 个 feature。为了简洁起见，我们就不一一列举了。
>
> 接下来，我们将问题表述为 pairwise 回归，并使用搜索 label 作为实用工具，并使用数据 $\mathcal{D}$ 来训练 GBDT 模型，并使用经过修改以支持 Lambda Rank 的软件包 4。 在离线评估不同模型时，我们在保留的搜索 session 集上使用 NDCG（标准的排序指标），即 $\mathcal{D}$ 的 80% 用于训练，而 20% 用于测试。
>
> 最终，模型经过训练后，即可用于搜索 listing 的在线评分。 用户 $u$ 执行的搜索查询 $q$ 返回的每个 listing 的特征向量  $\textbf{x}_i$ 都是线上计算的，并且使用分片架构并行进行评分。 在给定所有的分数后，按降序顺序向用户展示 listing 。
>
> 最后，一旦模型被训练，它被用于搜索 listing 的在线评分。对于用户 $u$ 执行的搜索查询 $q$ 返回的每个 listing，计算特征向量 $\textbf{x}_i$ 都是线上计算的，并且使用我们的分片架构并行评分 4。给定所有的分数后，将按照降序顺序向用户显示 listing。

#### Listing Embedding Features. 

The first step in adding embedding features to our Search Ranking Model was to load the 4.5 million embeddings into our search backend such that they can be accessed in real-time for feature calculation and model scoring.

Next, we introduced several user short-term history sets, that hold user actions from last 2 weeks, which are updated in real-time as new user actions happen. The logic was implemented using using Kafka 5 . Specifically, for each user_id we collect and maintain (regularly update) the following sets of listing ids:

1. $H_c$ : **clicked listing_ids** - listings that user clicked on in last 2 weeks. 
2. $H_{l_c}$ : **long-clicked listing_ids** - listing that user clicked and stayed on the listing page for longer than 60 sec. 
3. $H_s$ : **skipped listing_ids** - listings that user skipped in favor of a click on a lower positioned listing 
4. $H_w$ : **wishlisted listing_ids** - listings that user added to a wishlist in last 2 weeks. 
5. $H_i$ : **inquired listing_ids** - listings that user contacted in last 2 weeks but did not book. 
6. $H_b$ : **booked listing_ids** - listings that user booked in last 2 weeks.

We further split each of the short-term history sets $H_∗$ into subsets that contain listings from the same market. For example, if user had clicked on listings from New York and Los Angeles, their set $H_c$ would be further split into $H_c(NY)$ and $H_c(LA)$.

Finally, we define the embedding features which utilize the defined sets and the listing embeddings to produce a score for each candidate listing. The features are summarized in Table 6.

In the following we describe how $EmbClickSim$ feature is computed using $H_c$ . The rest of the features from top rows of Table 6 are computed in the same manner using their corresponding user short-term history set $H_∗$.

To compute $EmbClickSim$ for candidate listing $l_i$ we need to compute cosine similarity between its listing embedding $v_{l_i}$ and embeddings of listings in $H_c$. We do so by first computing $H_c$ market-level centroid embeddings. To illustrate, let us assume $H_c$ contains 5 listings from NY and 3 listings from LA. This would entail computing two market-level centroid embeddings, one for NY and one for LA, by averaging embeddings of listing ids from each of the markets. Finally, $EmbClickSim$ is calculated as maximum out of two similarities between listing embedding $\textbf{v}_{l_i}$ and $H_c$ market-level centroid embeddings.

More generally $EmbClickSim$ can be expressed as
$$
EmbClickSim(l_i,H_c) = \mathop{max}_{m \in M} 
\
cos(\textbf{v}_{l_i},\sum_{l_h \in m, l_h \in H_c}\textbf{v}_{l_h})

\qquad (10)
$$
where $M$ is the set of markets user had clicks in.

In addition to similarity to all user clicks, we added a feature that measures similarity to the latest long click, $EmbLastLongClickSim$. For a candidate listing $l_i$ it is calculated by finding the cosine similarity between its embedding $\textbf{v}_{l_i}$ and the embedding of the latest long clicked listing $l_{last}$ from $H_{l_c}$,
$$
EmbLastLongClickSim(l_i,H_{l_c})=cos(\textbf{v}_{l_i},\textbf{v}_{l_{last}})
\qquad (11)
$$

> 在我们的搜索排序模型中添加 embedding feature 的第一步是将 450万个 embedding feature 加载到我们的搜索后端，这样它们就可以实时访问，进行特征计算和模型评分。
>
> 接下来，我们介绍了几个用户历史 short-term 集合，它们保存了最近两周的用户行为，当新的用户行为发生时，它们会实时更新。该逻辑是使用 Kafka 5 实现的。具体来说，对于每个 user_id，我们收集并维护 (定期更新) 以下一组 listing id:
>
> 1. $H_c$：**clicked listing_ids** -用户在过去2周点击的 listings。
> 2. $H_{l_c}$：**long-clicked listing_ids** -用户点击并在 listing 页面上停留超过60秒 的 listing。
> 3. $H_s$：**skipped listing_ids** -用户跳过而选择点击排名较低的 listing 的 listing。
> 4. $H_w$：**wishlists listing_ids** - 用户过去两周内添加到 wishlist 的 listings。
> 5. $H_i$：**queries listing_ids**——用户在过去两周内联系但没有预订的 listing 。
> 6. $H_b$ : **booked listing_ids** -用户在过去两周预订的 listing。
>
> 我们进一步将每个 short-term 集合 $H_∗$ 分成同一市场的子集。例如，如果用户点击了来自纽约和洛杉矶的 listing，它们的 $H_c$ 将进一步分为 $H_c(NY)$ 和$H_c(LA)$。
>
> 最后，我们定义了 embedding 特征，这些特征利用定义的好的集合 和 listing embedding 为每个候选 listing 生成分数。 表6 中汇总了这些 feature。
>
> 下面我们将描述 $EmbClickSim$ 特征是如何使用 $H_c$ 计算的。使用相应的用户短期历史记录集 $H_*$，以相同的方式计算 表6 顶部各行中的其余 feature。
>
> 为了计算候选 listing $l_i$ 的 $EmbClickSim$，我们需要计算其 listing embedding $v_{l_i}$与 $H_c$ 中 listing embedding 之间的余弦相似度。为此，我们首先计算 $H_c$ 在市场级别上的 centroid embedding。为了说明问题，我们假设 $H_c$ 包含来自 NY 的 5 个 listing 和来自 LA 的 3 个 listing。这将需要通过计算两个市场级别的 centroid embedding，一个用于 NY(纽约)，一个用于 LA(洛杉矶)。
>
> 最后，计算 $EmbClickSim$ 为 listing embedding $\textbf{v}_{l_i}$ 与 $H_c$ 市场级别的 centroid embedding 的两个点中的最相似的。
>
> $EmbClickSim$  表示为：
> $$
> EmbClickSim(l_i,H_c) = \mathop{max}_{m \in M} 
> \
> cos(\textbf{v}_{l_i},\sum_{l_h \in m, l_h \in H_c}\textbf{v}_{l_h})
> 
> \qquad (10)
> $$
> 其中 $M$ 是用户曾经点击过的市场集合
>
> 除了与所有用户点击的相似性外，我们还添加了一个 feature，用于测量与latest long click 的相似性，即 $EmbLastLongClickSim$。对于候选 listing $l_i$，通过找出其 embedding $\textbf{v}_{l_i}$ 与来自 $H_{l_c}$ 的 latest long click $l_{last}$ 的 embedding 之间的余弦相似性来计算，
> $$
> EmbLastLongClickSim(l_i,H_{l_c})=cos(\textbf{v}_{l_i},\textbf{v}_{l_{last}})
> \qquad (11)
> $$

![Table6](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table6.png)

**Table 6: Embedding Features for Search Ranking**

#### User-type & Listing-type Embedding Features.

We follow similar procedure to introduce features based on user type and listing type embeddings. We trained embeddings for $500K$ user types and $500K$ listing types using $50$ million user booking sessions. Embeddings were $d = 32$ dimensional and were trained using a sliding window of $m = 5$ over booking sessions. The user type and listing type embeddings were loaded to search machines memory, such that we can compute the type similarities online.

To compute the $UserTypeListingTypeSim$ feature for candidate listing $l_i$ we simply look-up its current listing type $l_t$ as well as current user type $u_t$ of the user who is conducting the search and calculate cosine similarity between their embeddings,
$$
UserTypeListingTypeSim(u_t,l_t) = cos(\textbf{v}_{u_t},\textbf{v}_{l_t})
\qquad (12)
$$
All features from Table 6 were logged for $30$ days so they could be added to search ranking training set $\mathcal{D}$. The coverage of features, meaning the proportion of $\mathcal{D}$ which had particular feature populated, are reported in Table 7. As expected, it can be observed that features based on user clicks and skips have the highest coverage.

Finally, we trained a new GBDT Search Ranking model with embedding features added. Feature importances for embedding features (ranking among 104 features) are shown in Table 7. 

Top ranking features are similarity to listings user clicked on ($EmbClickSim$: ranked 5th overall) and similarity to listings user skipped ($EmbSkipSim$: ranked 8th overall). Five embedding features ranked among the top 20 features. As expected, long-term feature $UserTypeListingTypeSim$ which used all past user bookings ranked better than short-term feature $EmbBookSim$ which takes into account only bookings from last 2 weeks. This also shows that recommendations based on past bookings are better with embeddings that are trained using historical booking sessions instead of click sessions.

To evaluate if the model learned to use the features as we intended, we plot the partial dependency plots for 3 embedding features: $EmbClickSim$, $EmbSkipSim$ and $UserTypeListTypeSim$. These plots show what would happen to listing’s ranking score if we fix values of all but a single feature (the one we are examining). On the left subgraph it can be seen that large values of $EmbClickSim$, which convey that listing is similar to the listings user recently click on, lead to a higher model score. The middle subgraph shows that large values of $EmbSkipSim$, which indicate that listing is similar to the listings user skipped, lead to a lower model score. Finally, the right subgraph shows that large values of $UserTypeListingTypeSim$, which indicate that user type is similar to listing type, lead to a higher model score as expected.

> 我们遵循类似的过程来引入基于 user type 和 listing type 的embedding feature。我们使用 5000万 用户预订 session 训练了 $500K$ user type 和 500K  listing type 的 embedding。embedding 是 $d=32$ 维的，并且在预订过程中使用 $m=5$ 的滑动窗口进行训练。user type 和 listing type 的 embedding 被加载到搜索机内存中，以便我们可以在线计算类型相似度。
>
> 为了计算候选 listing $l_i$ 的 $UserTypeListingTypeSim$ 特征，我们简单地查找其当前 listing type $l_t$ 以及执行搜索的用户的当前 user type $u_t$ 并计算其 embedding之间的余弦相似性，
> $$
> UserTypeListingTypeSim(u_t,l_t) = cos(\textbf{v}_{u_t},\textbf{v}_{l_t})
> \qquad (12)
> $$
> 表6 中的所有 feature 都被记录 $30$ 天，这样它们就可以添加到搜索排序的训练集 $\mathcal{D}$ 中。表7 报告了 feature 的覆盖范围，即具有特定 feature 的 $\mathcal{D}$ 的比例。正如预期的那样，可以观察到基于用户 click 和 skip 的 feature 覆盖率最高。
>
> 最后，通过添加 embedding feature 训练出新的 GBDT 搜索排序模型。embedding 特征的特征重要性(104个特征中的排名)如表7所示。
>
> 排名最高的 feature 是 listings user clicked的相似度($EmbClickSim$：总体排名第5)，和 listings user skiped 的相似度（$ EmbSkipSim $：总体排名第8）。 前 20 个 feature 中有 5 个 embedding features。不出所料，长期 feature $UserTypeListingTypeSim$ 使用了所有过去的用户预订，其排名比短期 feature $ EmbBookSim $更好，后者仅考虑了最近两周的预订。这还表明基于过去预订的推荐能够更好地使用使用历史预订 session 训练的 embedding 而不是点击 session 训练的 embedding(注：这里是从特征重要性角度来讲的)。
>
> 为了评估模型是否学会了按照我们的预期使用这些特征，我们绘制了 3 个 embedding fatures 的 PDP :$EmbClickSim$、$EmbSkipSim$和$UserTypeListTypeSim$。
>
> 这些图显示了如果我们固定除了我们正要查看的 feature以外的所有 feature 的值，将会对 listing 的排序得分产生什么影响。
>
> 在左边的子图中，我们可以看到 $EmbClickSim$ 的值(表示 listing 与用户最近点击的 listing 相似度)越大模型得分越高。
>
> 在中间的子图中，$EmbSkipSim$ 的值(表示 listing 与用户跳过的 listing 的相似度)越大，模型得分越低。
>
> 右边的子图中，$UserTypeListingTypeSim$ 的值(user 类型与 listing 类型的相似度)越高，模型得分越高。

![Table7](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table7.png)

**Table 7: Embedding Features Coverage and Importances**

![Figure7](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig7.png)

**Figure 7: Partial Dependency Plots for EmbClickSim, EmbSkipSim and UserTypeListTypeSim**

#### Online Experiment Results Summary.

We conducted both offline and online experiments (A/B test). First, we compared two search ranking models trained on the same data with and without embedding features. In Table 8 we summarize the results in terms of DCU (Discounted Cumulative Utility) per each utility (impression, click, rejection and booking) and overall NDCU (Normalized Discounted Cumulative Utility). It can be observed that adding embedding features resulted in 2.27% lift in NDCU, where booking DCU increased by 2.58%, meaning that booked listings were ranked higher in the hold-out set, without any hit on rejections (DCU -0.4 was flat), meaning that rejected listings did not rank any higher than in the model without embedding features.

Observations from Table 8, plus the fact that embedding features ranked high in GBDT feature importances (Table 7) and the finding that features behavior matches what we intuitively expected (Figure 7) was enough to make a decision to proceed to an online experiment. In the online experiment we saw a statistically significant booking gain and embedding features were launched to production. Several months later we conducted a back test in which we attempted to remove the embedding features, and it resulted in negative bookings, which was another indicator that the real-time embedding features are effective.

> 我们进行了离线和在线实验(A / B测试)。 首先，我们比较了在具有和不具有 embedding feature 的情况下针对相同数据训练的两个搜索排序模型。 在表 8中，我们根据每个应用(impression(应该指的是曝光)，click，rejection和 booking) 的 DCU (Discounted Cumulative Utility) 和整体 NDCU（Normalized Discounted Cumulative Utility）汇总了结果。 
>
> 可以看出，加入 embedding feature使 NDCU 提升了 2.27%，booking DCU 增长了 2.58%，这意味着 booking listing 在 hold-out 集合中排名更高，没有任何 rejection (DCU -0.4 是持平的) ，这意味着被拒绝的 listing 不会比没有 embedding 特征的模型中的排名更高。
>
> 从表8中的观察结果，再加上 embedding feature 在 GBDT 特征重要性排名靠前的事实(表7)，以及发现 feature 的作用符合我们直观的预期(图7)，足以决定继续进行线上实验。线上实验中，我们看到了在 booking 上的明显收益，embedding feature 也开始投入生产中。几个月后，我们进行了一次反向测试，删除 embedding feature，结果导致了 booking 的负面影响，这是实时 embedding feature 有效的另一个指标。

![Table8](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table8.png)

**Table 8: Offline Experiment Results**

## 5 CONCLUSION

We proposed a novel method for real-time personalization in Search Ranking at Airbnb. The method learns low-dimensional representations of home listings and users based on contextual co-occurrence in user click and booking sessions. To better leverage available search contexts, we incorporate concepts such as global context and explicit negative signals into the training procedure. We evaluated the proposed method in Similar Listing Recommendations and Search Ranking. After successful test on live search traffic both embedding applications were deployed to production.

> 我们提出了一种在Airbnb搜索排序中进行实时个性化的新方法。该方法基于用户 click 和 bokking session 中的上下文共现来学习主页 listing 和用户的低维表示。为了更好地利用可用的搜索 context，我们将全局 context 和显式 negative signal 等概念纳入训练流程。我们在相似的 listing 推荐和搜索排序中对该方法进行了评估。在对实时搜索流量进行成功测试后，两个 embedding 应用程序都被部署到生产环境中。

## ACKNOWLEDGEMENTS

We would like to thank the entire Airbnb Search Ranking Team for their contributions to the project, especially Qing Zhang and Lynn Yang. We would also like to thank Phillippe Siclait and Matt Jones for creating the Embedding Evaluation Tool. The summary of this paper was published in Airbnb’s Medium Blog 6 .

> 我们要感谢整个 Airbnb 搜索排序团队为这个项目做出的贡献，特别是 Qing Zhang 和 Lynn Yang。我们还要感谢 Phillippe Sinlait 和 Matt Jones 创建了 embedding 评估工具。这篇论文的摘要发表在Airbnb的媒体博客6上。

## REFERENCES

[1] Kamelia Aryafar, Devin Guillory, and Liangjie Hong. 2016. An Ensemble-based Approach to Click-Through Rate Prediction for Promoted Listings at Etsy. In arXiv preprint arXiv:1711.01377. 

[2] Ricardo Baeza-Yates, Berthier Ribeiro-Neto, et al. 1999. Modern information retrieval. Vol. 463. ACM press New York. 

[3] Oren Barkan and Noam Koenigstein. 2016. Item2vec: neural item embedding for collaborative filtering. In Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on. IEEE, 1–6. 

[4] Christopher J Burges, Robert Ragno, and Quoc V Le. 2011. Learning to rank with nonsmooth cost functions. In Advances in NIPS 2007. 

[5] Ting Chen, Liangjie Hong, Yue Shi, and Yizhou Sun. 2017. Joint Text Embedding for Personalized Content-based Recommendation. In arXiv preprint arXiv:1706.01084. 

[6] Nemanja Djuric, Vladan Radosavljevic, Mihajlo Grbovic, and Narayan Bhamidipati. 2014. Hidden conditional random fields with distributed user embeddings for ad targeting. In IEEE International Conference on Data Mining. 

[7] Nemanja Djuric, Hao Wu, Vladan Radosavljevic, Mihajlo Grbovic, and Narayan Bhamidipati. 2015. Hierarchical neural language models for joint representation of streaming documents and their content. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 248–255.

[8] Mihajlo Grbovic, Nemanja Djuric, Vladan Radosavljevic, Fabrizio Silvestri, Ricardo Baeza-Yates, Andrew Feng, Erik Ordentlich, Lee Yang, and Gavin Owens. 2016. Scalable semantic matching of queries to ads in sponsored search advertising. In SIGIR 2016. ACM, 375–384. 

[9] Mihajlo Grbovic, Nemanja Djuric, Vladan Radosavljevic, Fabrizio Silvestri, and Narayan Bhamidipati. 2015. Context-and content-aware embeddings for query rewriting in sponsored search. In SIGIR 2015. ACM, 383–392. 

[10] Mihajlo Grbovic, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati, and Ananth Nagarajan. 2015. Gender and interest targeting for sponsored post advertising at tumblr. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1819–1828. 

[11] Mihajlo Grbovic, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati, Jaikit Savla, Varun Bhagwan, and Doug Sharp. 2015. E-commerce in your inbox: Product recommendations at scale. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 

[12] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 855–864. 

[13] Krishnaram Kenthapadi, Benjamin Le, and Ganesh Venkataraman. 2017. Personalized Job Recommendation System at LinkedIn: Practical Challenges and Lessons Learned. In Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 346–347. 

[14] Maciej Kula. 2015. Metadata embeddings for user and item cold-start recommendations. arXiv preprint arXiv:1507.08439 (2015). 

[15] Benjamin Le. 2017. Deep Learning for Personalized Search and Recommender Systems. In Slideshare: https://www.slideshare.net/BenjaminLe4/deep-learning-forpersonalized-search-and-recommender-systems. 

[16] Steve Liu. 2017. Personalized Recommendations at Tinder: The TinVec Approach. In Slideshare: https://www.slideshare.net/SessionsEvents/dr-steve-liu-chief-scientisttinder-at-mlconf-sf-2017. 

[17] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems. 3111–3119. 

[18] Thomas Nedelec, Elena Smirnova, and Flavian Vasile. 2017. Specializing Joint Representations for the task of Product Recommendation. arXiv preprint arXiv:1706.07625 (2017). 

[19] Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017. Embedding-based news recommendation for millions of users. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1933–1942. 

[20] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 701–710. 

[21] Vladan Radosavljevic, Mihajlo Grbovic, Nemanja Djuric, Narayan Bhamidipati, Daneo Zhang, Jack Wang, Jiankai Dang, Haiying Huang, Ananth Nagarajan, and Peiji Chen. 2016. Smartphone app categorization for interest targeting in advertising marketplace. In Proceedings of the 25th International Conference Companion on World Wide Web. International World Wide Web Conferences Steering Committee, 93–94. 

[22] Sharath Rao. 2017. Learned Embeddings for Search at Instacart. In Slideshare: https://www.slideshare.net/SharathRao6/learned-embeddings-for-searchand-discovery-at-instacart. 

[23] Thomas Schmitt, François Gonard, Philippe Caillou, and Michèle Sebag. 2017. Language Modelling for Collaborative Filtering: Application to Job Applicant Matching. In IEEE International Conference on Tools with Artificial Intelligence. 

[24] Yukihiro Tagami, Hayato Kobayashi, Shingo Ono, and Akira Tajima. 2015. Modeling User Activities on the Web using Paragraph Vector. In Proceedings of the 24th International Conference on World Wide Web. ACM, 125–126. 

[25] Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: a simple and general method for semi-supervised learning. In Proceedings of the 48th annual meeting of the association for computational linguistics. Association for Computational Linguistics, 384–394.

[26] Dongjing Wang, Shuiguang Deng, Xin Zhang, and Guandong Xu. 2016. Learning music embedding with metadata for context aware recommendation. In Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval. 

[27] Jason Weston, Ron J Weiss, and Hector Yee. 2013. Nonlinear latent factorization by embedding multiple user interests. In Proceedings of the 7th ACM conference on Recommender systems. ACM, 65–68. 

[28] Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, and Jason Weston. 2017. StarSpace: Embed All The Things! arXiv preprint arXiv:1709.03856. 

[29] Dawei Yin, Yuening Hu, Jiliang Tang, Tim Daly, Mianwei Zhou, Hua Ouyang, Jianhui Chen, Changsung Kang, Hongbo Deng, Chikashi Nobata, et al. 2016. Ranking relevance in yahoo search. In Proceedings of the 22nd ACM SIGKDD.

