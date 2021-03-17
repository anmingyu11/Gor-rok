# Real-time Personalization using Embeddings for Search Ranking at Airbnb

## ABSTRACT

Search Ranking and Recommendations are fundamental problems of crucial interest to major Internet companies, including web search engines, content publishing websites and marketplaces. However, despite sharing some common characteristics a one-size-fitsall solution does not exist in this space. Given a large difference in content that needs to be ranked, personalized and recommended, each marketplace has a somewhat unique challenge. 

Correspondingly, at Airbnb, a short-term rental marketplace, search and recommendation problems are quite unique, being a two-sided marketplace in which one needs to optimize for host and guest preferences, in a world where a user rarely consumes the same item twice and one listing can accept only one guest for a certain set of dates. 

In this paper we describe Listing and User Embedding techniques we developed and deployed for purposes of Real-time Personalization in Search Ranking and Similar Listing Recommendations, two channels that drive 99% of conversions. The embedding models were specifically tailored for Airbnb marketplace, and are able to capture guest’s short-term and long-term interests, delivering effective home listing recommendations. We conducted rigorous offline testing of the embedding models, followed by successful online tests before fully deploying them into production.

> 搜索排名和建议是多数 互联网公司（包括Web搜索引擎，内容发布网站和市场）至关重要的基本问题。 但是，尽管具有一些共同性，但在这个领域中不存在“一刀切”的解决方案。 鉴于需要对内容进行 排序，个性化和推荐的差异很大，因此每个市场都面临着一些独特的挑战。
>
> 相应地，在 Airbnb 这个短租市场上，搜索和推荐问题非常独特，这是一个双向市场，在这个市场中，用户需要针对 host 和 guest 的喜好进行优化，在这里，用户很少会两次消费相同的商品,且一个房源在特定的日期只能接受一个客人。
>
> 在本文中，我们将描述 为实时个性化而开发和部署的 Listing 和 User Embedding  技术，它们用于 “搜索排序” 和 “相似列表推荐”中，这两个渠道可带来 99％的转化。
>
> embedding 模型是为 Airbnb的市场 量身定做的，能够捕捉客人的短期和长期兴趣，提供有效的房屋列表推荐。
>
> 我们对 embedding 模型进行了严格的离线测试，然后在将它们完全部署到生产环境之前进行了成功的在线测试。

## CCS CONCEPTS

Information systems → Content ranking; Web log analysis; Personalization; Query representation; Document representation;

## KEYWORDS

Search Ranking; User Modeling; Personalization

## 1 INTRODUCTION

During last decade Search architectures, which were typically based on classic Information Retrieval, have seen an increased presence of Machine Learning in its various components [2], especially in Search Ranking which often has challenging objectives depending on the type of content that is being searched over. The main reason behind this trend is the rise in the amount of search data that can be collected and analyzed. The large amounts of collected data open up possibilities for using Machine Learning to personalize search results for a particular user based on previous searches and recommend similar content to recently consumed one.

The objective of any search algorithm can vary depending on the platform at hand. While some platforms aim at increasing website engagement (e.g. clicks and time spent on news articles that are being searched), others aim at maximizing conversions (e.g. purchases of goods or services that are being searched over), and in the case of two sided marketplaces we often need to optimize the search results for both sides of the marketplace, i.e. sellers and buyers. 

The two sided marketplaces have emerged as a viable business model in many real world applications. In particular, we have moved from the social network paradigm to a network with two distinct types of participants representing supply and demand. Example industries include accommodation (Airbnb), ride sharing (Uber, Lyft), online shops (Etsy), etc. Arguably, content discovery and search ranking for these types of marketplaces need to satisfy both supply and demand sides of the ecosystem in order to grow and prosper.

In the case of Airbnb, there is a clear need to optimize search results for both hosts and guests, meaning that given an input query with location and trip dates we need to rank high listings whose location, price, style, reviews, etc. are appealing to the guest and, at the same time, are a good match in terms of host preferences for trip duration and lead days. 

Furthermore, we need to detect listings that would likely reject the guest due to bad reviews, pets, length of stay, group size or any other factor, and rank these listings lower. To achieve this we resort to using Learning to Rank. Specifically, we formulate the problem as pairwise regression with positive utilities for bookings and negative utilities for rejections, which we optimize using a modified version of Lambda Rank [4] model that jointly optimizes ranking for both sides of the marketplace.

Since guests typically conduct multiple searches before booking, i.e. click on more than one listing and contact more than one host during their search session, we can use these in-session signals, i.e. clicks, host contacts, etc. for Real-time Personalization where the aim is to show to the guest more of the listings similar to the ones we think they liked since staring the search session. 

At the same time we can use the negative signal, e.g. skips of high ranked listings, to show to the guest less of the listings similar to the ones we think they did not like. To be able to calculate similarities between listings that guest interacted with and candidate listings that need to be ranked we propose to use listing embeddings, low-dimensional vector representations learned from search sessions. We leverage these similarities to create personalization features for our Search Ranking Model and to power our Similar Listing Recommendations, the two platforms that drive $99\%$ of bookings at Airbnb.

> 在过去的十年中，通常以经典信息检索为基础的搜索架构，在其各个组件[2]中看到了越来越多的机器学习的存在，特别是在搜索排序中，根据所搜索的内容类型不同，通常极有挑战性。这一趋势背后的主要原因是可以收集和分析的搜索数据量的增加。大量的数据为使用机器学习的实践提供了可能性，包括根据之前的搜索记录去个性化用户的搜索结果，或者通过最近的消费记录去推荐相似内容。
>
> 任何搜索算法的目标都可以根据当前平台的不同而有所不同。 尽管某些平台旨在提高网站的参与度（例如，在被搜索的新闻文章上的点击量和时间），而其他平台则旨在最大程度地提高转化率（例如，购买被搜索的商品或服务），以及在双向市场中 我们经常需要针对市场双方（即卖方和买方）优化搜索结果。
>
> 在许多现实世界的应用程序中，双向市场已经成为一种可行的商业模式。特别的，我们已经从社交网络转变为具有代表供应和需求的两种不同类型参与者的网络。示例行业包括住宿(Airbnb)、拼车(Uber、Lyft)、在线商店(Etsy)等。可以说，这些类型市场的内容发现和搜索排序需要满足生态的供给方和需求方，才能增长和繁荣。
>
> 对于 Airbnb 而言，显然需要针对 hosts 和 guests 优化搜索结果，这意味着，在输入查询中提供位置和行程日期后，我们需要对位置、价格、风格、评论等对 guests 有吸引力的 listings 进行排序，同时需要在旅行持续时间和提前天数方面与 hosts 的偏好很好地匹配。
>
> 此外，我们还需要检测那些可能因为评论不佳、宠物、住宿时间、团体人数或任何其他因素而可能拒绝客人的房源，并将这些房源的排名降低。为了实现这一点，我们使用“学习排序”。具体来说，我们将问题表述为成对回归，其中预订为正效用和拒绝为负效用，我们使用 Lambda Rank [4]模型的改进版进行优化，该模型同时优化了市场双方的排序。
>
> 由于 guests 通常在预订前进行多次搜索，即在搜索 session 期间点击一个以上的房源并联系多个房东，因此我们可以使用这些会话中的 signal (即点击、房东联系人等)进行实时个性化，其目的是向 guests 展示更多自他们的搜索 session 开始以来与我们认为他们喜欢的列表的相似列表(注：很拗口，我在经验上理解的意思应该是这里假设用户点击或者联系的房源是客人感兴趣的，认为这些是他们喜欢的房源，从这些基础上去推荐相似的房源。)
>
> 同时，我们可以使用负信号，例如跳过排名较高的列表，向 guests 较少地展示与我们认为他们不喜欢的  listing 相似的 listing。为了能够计算 guest 与之交互的 listing 和需要排序的候选 listing 之间的相似性，我们建议使用 listing embedding，即从搜索 session 中学习低维向量表示。我们利用这些相似性为我们的搜索排序模型创建个性化功能，并助力我们的相似列表推荐，这两个平台推动了 Airbnb $99\%$ 的预订量。

In addition to Real-time Personalization using immediate user actions, such as clicks, that can be used as proxy signal for shortterm user interest, we introduce another type of embeddings trained on bookings to be able to capture user’s long-term interest. Due to the nature of travel business, where users travel 1-2 times per year on average, bookings are a sparse signal, with a long tail of users with a single booking. To tackle this we propose to train embeddings at a level of user type, instead of a particular user id, where type is determined using many-to-one rule-based mapping that leverages known user attributes. 

At the same time we learn listing type embeddings in the same vector space as user type embeddings. This enables us to calculate similarities between user type embedding of the user who is conducting a search and listing type embeddings of candidate listings that need to be ranked. 

Compared to previously published work on embeddings for personalization on the Web, novel contributions of this paper are:

- **Real-time Personalization** - Most of the previous work on personalization and item recommendations using embeddings [8, 11] is deployed to production by forming tables of user-item and item-item recommendations offline, and then reading from them at the time of recommendation. We implemented a solution where embeddings of items that user most recently interacted with are combined in an online manner to calculate similarities to items that need to be ranked.
- **Adapting Training for Congregated Search** - Unlike in Web search, the search on travel platforms is often congregated, where users frequently search only within a certain market, e.g. Paris., and rarely across different markets. We adapted the embedding training algorithm to take this into account when doing negative sampling, which lead to capturing better within-market listings similarities.
- **Leveraging Conversions as Global Context** - We recognize the importance of click sessions that end up in conversion, in our case booking. When learning listing embeddings we treat the booked listing as global context that is always being predicted as the window moves over the session.
- **User Type Embeddings** - Previous work on training user embeddings to capture their long-term interest [6, 27] train a separate embedding for each user. When target signal is sparse, there is not enough data to train a good embedding representation for each user. Not to mention that storing embeddings for each user to perform online calculations would require lot of memory. For that reason we propose to train embeddings at a level of user type, where groups of users with same type will have the same embedding.
- **Rejections as Explicit Negatives** - To reduce recommendations that result in rejections we encode host preference signal in user and listing type embeddings by treating host rejections as explicit negatives during training.

For short-term interest personalization we trained listing embeddings using more than 800 million search clicks sessions, resulting in high quality listing representations. We used extensive offline and online evaluation on real search traffic which showed that adding embedding features to the ranking model resulted in significant booking gain. In addition to the search ranking algorithm, listing embeddings were successfully tested and launched for similar listing recommendations where they outperformed the existing algorithm click-through rate (CTR) by 20%.

For long-term interest personalization we trained user type and listing type embeddings using sequences of booked listings by 50 million users. Both user and listing type embeddings were learned in the same vector space, such that we can calculate similarities between user type and listing types of listings that need to be ranked. The similarity was used as an additional feature for search ranking model and was also successfully tested and launched.

> 除了使用可用作短期用户兴趣的代理信号的即时用户动作(如点击)进行实时个性化之外，我们还引入了另一种针对预订的 embedding，以能够捕获用户的长期兴趣。由于旅游业务的性质，用户平均每年旅行1-2次，预订是一个稀疏的信号，预订了一次的用户是长尾。为了解决这个问题，我们建议在用户类型层面而不是特定的用户 ID 上训练 embedding，其中类型是使用利用已知用户属性的多对一基于规则的映射来确定的。
>
> 同时，我们在与 user 类型 embedding 相同的向量空间中学习 listing 类型embedding。这使我们能够计算正在进行搜索的用户的用户类型 embedding 和需要排序的候选 listing 的 listing 类型 embedding 之间的相似性。与以前发布的有关个性化 embedding 的工作相比，本本文的创新之处在于：
>
> - **Real-time Personalization** -以前的大多数关于个性化和 item 推荐的工作都是使用embeddings [8,11]，通过生成 user-item和 item-item 的离线推荐表，然后在推荐时读取它们来部署到生产。我们实现了一个解决方案，其中以在线方式组合了用户最近与之交互的 item 的 embedding，以计算与需要排序的 item 的相似度。
> - **Adapting Training for Congregated Search** - 与网页搜索不同的是，旅游平台上的搜索通常是聚集的，用户经常只在某个特定的市场进行搜索，例如巴黎。很少跨不同的市场进行搜索。我们对 embedding 训练算法进行了调整，以在进行 negative sampling 时考虑到这一点，从而可以更好地捕获市场内 listing 的相似性。
> - **Leveraging Conversions as Global Context** - 我们意识到了点击 session 最终转化的重要性，当学习 listing embedding 是，我们将预订的 listing 视为全局上下文，当窗口在 session 中移动时，它总是会被预测。
> - **User Type Embeddings** -以前的工作是训练用户 embedding 来捕捉他们的长期兴趣[6,27]，为每个用户训练单独的 embedding。 当目标信号稀疏时，没有足够的数据来训练每个用户的良好的 embedding 表示。 更不用说为每个用户存储 embedding 以执行在线计算将需要大量内存。 因此，我们建议在用户类型的层次上上训练 embedding，其中具有相同类型的用户组将具有相同的embedding。
> - **Rejections as Explicit Negatives** - 为了减少导致拒绝的推荐，我们在训练期间将 host 拒绝视为显式否定，在 user 和 listing 类型的 embeddings 中对 host 偏好进行编码。
>
> 对于短期兴趣个性化，我们使用超过8亿次搜索点击 session 来训练 listing embedding ，从而获得高质量的 listing 表示。 我们对真实搜索流量进行了广泛的离线和在线评估，结果表明，在排序模型中加入 embedding 特征可显着提高预订量。 除搜索排序算法外，还成功测试并上线了针对类似 listing 推荐的 listing embedding，其点击率(CTR)比现有算法高出20%。
>
> 对于长期兴趣个性化，我们使用 5000万 用户预订的序列来训练 user type 和listing type 的 embedding。user 和 listing type 的 embedding 都是在同一向量空间中学习的，这样我们就可以计算需要排序的 listings 的 user type 和 listing type 之间的相似性。相似度被用作搜索排序模型的 addtional feature ，并成功地进行了测试和上线。

## 2 RELATED WORK

In a number of Natural Language Processing (NLP) applications classic methods for language modeling that represent words as highdimensional, sparse vectors have been replaced by Neural Language models that learn word embeddings, i.e. low-dimensional representations of words, through the use of neural networks [25, 27]. The networks are trained by directly taking into account the word order and their co-occurrence, based on the assumption that words frequently appearing together in the sentences also share more statistical dependence. With the development of highly scalable continuous bag-of-words (CBOW) and skip-gram (SG) language models for word representation learning [17], the embedding models have been shown to obtain state-of-the-art performance on many traditional language tasks after training on large text data.

More recently, the concept of embeddings has been extended beyond word representations to other applications outside of NLP domain. Researchers from the Web Search, E-commerce and Marketplace domains have quickly realized that just like one can train word embeddings by treating a sequence of words in a sentence as context, same can be done for training embeddings of user actions, e.g. items that were clicked or purchased [11, 18], queries and ads that were clicked [8, 9], by treating sequence of user actions as context. 

Ever since, we have seen embeddings being leveraged for various types of recommendations on the Web, including music recommendations [26], job search [13], app recommendations [21], movie recommendations [3, 7], etc. Furthermore, it has been shown that items which user interacted with can be leveraged to directly lean user embeddings in the same feature space as item embeddings, such that direct user-item recommendations can be made [6, 10, 11, 24, 27]. 

Alternative approach, specifically useful for cold-start recommendations, is to still to use text embeddings (e.g. ones publicly available at https://code.google.com/p/word2vec) and leverage item and/or user meta data (e.g. title and description) to compute their embeddings [5, 14, 19, 28]. Finally, similar extensions of embedding approaches have been proposed for Social Network analysis, where random walks on graphs can be used to learn embeddings of nodes in graph structure [12, 20].

Embedding approaches have had a major impact in both academia and industry circles. Recent industry conference publications and talks show that they have been successfully deployed in various personalization, recommendation and ranking engines of major Web companies, such as Yahoo [8, 11, 29], Etsy [1], Criteo [18], Linkedin [15, 23], Tinder [16], Tumblr [10], Instacart [22], Facebook [28].

> 在许多自然语言处理（NLP）应用中，经典的语言建模方法将词表示为高维稀疏向量的方法已被神经语言模型所取代，后者通过使用神经网络来学习词的 embedding（即单词的低维表示） [25，27]。 **网络的训练是通过考虑词序和它们的共现，基于一个假设，即经常一起出现在句子中的词也享有更多的统计依赖性。 随着用于单词表示学习的高度可扩展的 continueous-bag-of-words (CBOW)和 skip-gram (SG)语言模型的发展[17]，在对大量文本数据进行训练后，embedding 模型在许多传统语言任务上都获得了最先进的性能。
>
> 最近，embedding 的概念已经从词表示扩展到NLP领域之外的其他应用。来自网页搜索、电子商务和市场领域的研究人员很快意识到，就像人们可以通过将句子中的词序列视为上下文来训练词 embedding 入一样，可以通过将用户动作 action 视为上下文来训练用户 action 的 embedding，例如被点击或购买的 item [11，18]、被点击的查询和广告[8，9]。
>
> 从那以后，我们看到 embedding 功能被应用于各种类型的网络推荐，包括音乐推荐[26]、求职推荐[13]、应用推荐[21]、电影推荐[3,7]等。此外，已经显示出可以与用户交互的 item 被利用来在与 item embedding 相同的特征空间中直接向用户 embedding 学习，从而可以做出直接的用户 embedding 推荐[6、10、11、24、27]。
>
> 对于冷启动推荐特别有用的另一种方法是仍然使用 word embedding (例如，在https://code.google.com/p/word2vec)公开可用的 word embedding，并利用item 和/或 user meta data(例如，标题和描述)来计算它们 embedding [5、14、19、28]。最后，embedding 方法的类似扩展已经被提出用于 Social Network 分析，其中在图上的随机游走可以用来学习图结构中节点的embedding [12，20]。
>
> embedding 方法在学术界和产业界都产生了重大影响。最近的行业会议出版物和对话表明，它们已经成功地部署在主要网络公司的各种个性化、推荐和排序引擎中，如 Yahoo[8，11，29]，Etsy[1]，Criteo[18]，LinkedIn[15，23]，Tinder[16]，Tumblr[10]，Insta[22]，Facebook[28]。

## 3 METHODOLOGY（方法论）

In the following we introduce the proposed methodology for the task of listing recommendations and listing ranking in search at Airbnb. We describe two distinct approaches, i.e. listing embeddings for short-term real-time personalization and user-type & listing type embeddings for long term personalization, respectively.

> 下面，我们将介绍 Airbnb 推荐和搜索排序的方法。我们分别描述了两种不同的方法，即用于短期实时个性化的 listing embedding 和用于长期个性化的 user-type & listing type embedding。

#### 3.1 Listing Embeddings

Let us assume we are given a set $\mathcal{S}$ of $S$ click sessions obtained from $N$ users, where each session $s = (l_1,\cdots ,l_M ) \in \mathcal{S}$ is defined as an uninterrupted sequence of $M$ listing ids that were clicked by the user. A new session is started whenever there is a time gap of more than $30$ minutes between two consecutive user clicks. Given this data set, the aim is to learn a d-dimensional real-valued representation $v_{l_i }∈ \mathbb{R}^d$ of each unique listing $l_i$ , such that similar listings lie nearby in the embedding space.

More formally, the objective of the model is to learn listing representations using the skip-gram model [17] by maximizing the objective function $\mathcal{L}$ over the entire set $\mathcal{S}$ of search sessions, defined as follows
$$
\mathcal{L} = \sum_{s \in \mathcal{S}}\sum_{l_i \in s}
(\sum_{-m \ge j \le m, i \ne 0} log \ \mathbb{P}(l_{i+j} | l_i))
\qquad (1)
$$
Probability $\mathbb{P}(l_{i+j} |l_i)$ of observing a listing $l_{i+j}$ from the contextual neighborhood of clicked listing $l_i$ is defined using the soft-max
$$
\mathbb{P}(l_{i+j}|l_i) = \frac{exp(v_{l_i}^{T}v_{i+j}^{'})}{\sum_{l=1}^{|\mathcal{V}|}exp(v_{l_i}^{T}v_{l}^{'})}
\qquad(2)
$$
where $\textbf{v}_l$ and $\textbf{v}_l^{'}$ are the input and output vector representations of listing $l$, hyperparameter $m$ is defined as a length of the relevant forward looking and backward looking context (neighborhood) for a clicked listing, and $\mathcal{V}$ is a vocabulary defined as a set of unique listings ids in the data set. From (1) and (2) we see that the proposed approach models temporal context of listing click sequences, where listings with similar contexts (i.e., with similar neighboring listings in search sessions) will have similar representations.

Time required to compute gradient $\nabla \mathcal{L}$ of the objective function in (1) is proportional to the vocabulary size $|\mathcal{V}|$ , which for large vocabularies, e.g. several millions listing ids, is an infeasible task. As an alternative we used negative sampling approach proposed in [17], which significantly reduces computational complexity. Negative sampling can be formulated as follows. We generate a set $D_p$ of positive pairs $(l,c)$ of clicked listings $l$ and their contexts $c$ (i.e., clicks on other listings by the same user that happened before and after click on listing $l$ within a window of length $m$), and a set $D_n$ of negative pairs $(l,c)$ of clicked listings and $n$ randomly sampled listings from the entire vocabulary $\mathcal{V}$. The optimization objective then becomes:
$$
\mathop{argmax}_{\theta} =
\sum_{(l,c) \in \mathcal{D_p}}
\
log{\frac{1}{1+e^{-\textbf{v}_{c}^{'}\textbf{v}_l}}}
+
\sum_{(l,c) \in \mathcal{D}_n}
\
log{\frac{1}{1+e^{\textbf{v}_c{'}\textbf{v}_l}}}
\qquad (3)
$$
where parameters $θ$ to be learned are $v_l$ and $v_c$ , $l,c \in V$. The optimization is done via stochastic gradient ascent.



> 假设我们有一个 $\mathcal{S}$ 集合，其中 $S$个 单击 session 是从 $N$个 用户中获得的，其中 $\mathcal{S}$ 中的每个 session $s=(l_1，\cdots，l_M) \in \mathcal{S}$ 被定义为用户单击的 $M$ 个 listing ID 的 uninterrupted 的连续序列。当用户连续两次点击之间的时间间隔超过 $30$ minutes时，就会启动一个新的 session 。给定该数据集，目标是学习每个唯一 listing $ l_i $ 的 $d$ 维实值表示 $ v_{l_i} \in \mathbb{R} ^ d $，以便类似 listing 位于 embbeding 空间附近 。
>
> 形式化来讲，该模型的目标是通过在搜索 session集合 $\mathcal{S}$ 上最大化目标函数 $\mathcal{L}$ 来学习使用 skip-gram 模型[17]的 listing 表示，定义如下(原论文的这个公式似乎有处 typo? 这里放上原公式，实际上j就是表达在context window 内除 i 以外的 listing)
> $$
> \mathcal{L} = \sum_{s \in \mathcal{S}}\sum_{l_i \in s}
> (\sum_{-m \ge j \le m, i \ne 0} log \ \mathbb{P}(l_{i+j} | l_i))
> \qquad (1)
> $$
> 其中 $\textbf{v}_l$ 和 $\textbf{v}_l^{'}$ 是列表 $l$ 的输入和输出向量表示，超参数 $m$ 被定义为 clicked listing 的相关前后上下文(邻居)的长度，$\mathcal{V}$ 是 listing ID 的词典长度。从 (1) 和 (2) 我们可以看出，所提出的方法对 listing click 序列的时间上下文进行建模，其中具有相似上下文的 listing (即，在搜索 session 中具有类似的相邻 listing )将具有相似的表示。
>
> 计算(1)中的目标函数的梯度 $\nabla \mathcal{L}$所需的时间与词典大小  $|\mathcal{V}|$ 成正比，这对于较大的词典(例如数百万个 listing ID)来说是不可行的任务。作为另一种选择，我们使用了[17]中提出的 negative sampling 方法，这大大降低了计算复杂度。负抽样的形式化定义如下。
>
> 我们生成被点击的 listing $l$ 及其上下文 $c$ 的正样本对 $(l，c)$ 的集合 $D_p$ (即，在长度为 $m$ 的窗口内，同一用户在点击 listing $l$ 的前后点击的其他 listing )，以及被点击的 listing 和来自整个词汇表 $\mathcal{V}$ 的 $n$ 个随机抽样的列表的**负样本对** $(l，c)$ 的集合 $D_n$。
>
> 然后，优化目标变为：
> $$
> \mathop{argmax}_{\theta}
> \sum_{(l,c) \in \mathcal{D_p}}
> \
> log{\frac{1}{1+e^{-\textbf{v}_{c}^{'}\textbf{v}_l}}}
> +
> \sum_{(l,c) \in \mathcal{D}_n}
> \
> log{\frac{1}{1+e^{\textbf{v}_c{'}\textbf{v}_l}}}
> \qquad (3)
> $$
> 需要优化的参数 $\theta$ 是 $\textbf{v}_c$ 和  $\textbf{v}_l$ ，$l,c \in \mathcal{V}$. 使用随机梯度下降优化。



**Booked Listing as Global Context. **

We can break down the click sessions set $\mathcal{S}$ into 

1. booked sessions, i.e. click sessions that end with user booking a listing to stay at.
2. exploratory sessions, i.e. click sessions that do not end with booking, i.e. users were just browsing. 

Both are useful from the standpoint of capturing contextual similarity, however booked sessions can be used to adapt the optimization such that at each step we predict not only the neighboring clicked listings but the eventually booked listing as well. This adaptation can be achieved by adding booked listing as global context, such that it will always be predicted no matter if it is within the context window or not. Consequently, for booked sessions the embedding update rule becomes
$$
\mathop{argmax}_{\theta} 
\sum_{(l,c) \in \mathcal{D}_p}
log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_l}}
+
\sum_{(l,c) \in \mathcal{D}_n}
log \frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_l}}
+
log \frac{1}{1+e^{\textbf{-v}_{l_b}^{'}\textbf{v}_l}}
\\
\qquad (4)
$$
where $\textbf{v}_{l_b}$ is the embedding of the booked listing $l_b$ . For exploratory sessions the updates are still conducted by optimizing objective (3)

Figure 1 shows a graphical representation of how listing embeddings are learned from booked sessions using a sliding window of size $2n + 1$​ that slides from the first clicked listing to the booked listing. At each step the embedding of the central listing $\textbf{v}_l$ is being updated such that it predicts the embeddings of the context listings $\textbf{v}_c$ from $\mathcal{D}_p $and the booked listing $\textbf{v}_{l_b}$ . As the window slides some listings fall in and out of the context set, while the booked listing always remains within it as global context (dotted line)

> 我们可以将集合 $\mathcal{S}$ 的点击 session 分解为。
>
> 1.预订的会话，以预订为结束的 session.
> 2.探索性会话，即不以预订结束的点击 session，即用户只是浏览。
>
> 从获取上下文相似性的角度来看，这两者都是有用的，但是可以使用预订的 session 来进行优化，以便在每个步骤中，我们不仅可以预测相邻的点击 listing ，还可以预测最终的预订 listing。 这种优化可以通过添加预订 listing 作为全局上下文来实现，这样无论它是否在上下文窗口内，都将始终对其进行预测。 因此，对于预订的 session，embedding 更新规则变为：
> $$
> \mathop{argmax}_{\theta} 
> \sum_{(l,c) \in \mathcal{D}_p}
> log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_l}}
> +
> \sum_{(l,c) \in \mathcal{D}_n}
> log \frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_l}}
> +
> log \frac{1}{1+e^{\textbf{-v}_{l_b}^{'}\textbf{v}_l}}
> \\
> \qquad (4)
> $$
> 图1 显示了如何使用大小为 $2n+1$ 的滑动窗口从预订的 session 中学习 listing embedding，该滑动窗口从第一次点击的 listing 滑动到预订的 listing (注：从第一个到最后一个)。每一步，中心词 listing $\textbf{v}_l$ 的 embedding 都会被更新，使得它预测来自 $\mathcal{D}_p$ 的上下文 listing $\textbf{v}_c$ 和预订的 listing $\textbf{v}_{l_b}$ 的 embedding 。当窗口滑动时，一些 listing 在上下文集合中进进出出(注：上下文集合随着窗口滑动而改变)，而预订的 listing 始终作为全局上下文保留在其中(虚线)。

#### Adapting Training for Congregated Search

Users of online travel booking sites typically search only within a single market, i.e. location they want to stay at. As a consequence, there is a high probability that $\mathcal{D}_p$ contains listings from the same market. On the other hand, due to random sampling of negatives, it is very likely that $\mathcal{D}_n$ contains mostly listings that are not from the same markets as listings in $\mathcal{D}_p $. At each step, for a given central listing $l$, the positive context mostly consist of listings from the same market as $l$, while the negative context mostly consists of listings that are not from the same market as $l$. We found that this imbalance leads to learning sub-optimal within-market similarities. To address this issue we propose to add a set of random negatives $\mathcal{D}_{m_n}$ , sampled from the market of the central listing $l$.
$$
\mathop{argmax}_{\theta} 
\sum_{(l,c) \in \mathcal{D}_p}log \ \frac{1}{1 + e^{-\textbf{v}_c^{'}\textbf{v}_l}}
+
\sum_{(l,c) \in \mathcal{D}_n} log \frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_l}}
\\
+
log\frac{1}{1+e^{-\textbf{v}_{l_b}^{'}\textbf{v}_l}}
+
\sum_{(l,m_n) \in \mathcal{D}_{m_n}} log\frac{1}{1+e^{\textbf{v}_{m_n}^{'}\textbf{v}_l}}.

\qquad (5)
$$
where parameters $θ$ to be learned are $\textbf{v}_l$ and $\textbf{v}_c$ , $l,c \in \mathcal{V}$.

> 在线旅游预订网站的用户通常只在一个市场内搜索，即他们想要去的地儿。因此，$\mathcal{D}_p$ 大多是同一市场的 listing 。另一方面，由于对，$\mathcal{D}_n$ 的大多数 listing 与 $\mathcal{D}_p$中的 listing 来自不同的市场。在每一步骤中，对于给定的中心 listing l，正样本上下文主要由来自与 $l$ 相同市场的 listing 组成，而负样本上下文主要由来自与 $l$ 不同市场的 listing 组成。我们发现，这种不平衡导致学习了次优的市场内的 listing 的相似性。为了解决这个问题，我们建议添加一组随机的负样本集合 $\mathcal{D}_{m_n}$，从中心 listing $l$ 所属的市场中抽样。
> $$
> \mathop{argmax}_{\theta} 
> \sum_{(l,c) \in \mathcal{D}_p}log \ \frac{1}{1 + e^{-\textbf{v}_c^{'}\textbf{v}_l}}
> +
> \sum_{(l,c) \in \mathcal{D}_n} log \frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_l}}
> \\
> +
> log\frac{1}{1+e^{-\textbf{v}_{l_b}^{'}\textbf{v}_l}}
> +
> \sum_{(l,m_n) \in \mathcal{D}_{m_n}} log\frac{1}{1+e^{\textbf{v}_{m_n}^{'}\textbf{v}_l}}.
> 
> \qquad (5)
> $$
> 学习的参数是 $\textbf{v}_l$ 和 $\textbf{v}_c$ , 其中 $l,c \in \mathcal{V}$.

#### Cold start listing embeddings.

Every day new listings are created by hosts and made available on Airbnb. At that point these listings do not have an embedding because they were not present in the click sessions $\mathcal{S}$ training data. To create embeddings for new listings we propose to utilize existing embeddings of other listings. Upon listing creation the host is required to provide information about the listing, such as location, price, listing type, etc. We use the provided meta-data about the listing to find $3$ geographically closest listings (within a 10 miles radius) that have embeddings, are of same listing type as the new listing (e.g. Private Room) and belong to the same price bucket as the new listing (e.g. $ \$20 − \$25$ per night). Next, we calculate the mean vector using $3$ embeddings of identified listings to form the new listing embedding. Using this technique we are able to cover more than $98\%$ of new listings.

> 每天，房东都会在Airbnb上创建新的房源。此时，这些 listing 没有embedding，因为它们没有出现在点击 session $\mathcal{S}$ 训练数据中。为了创建冷启动的 listing 的 embedding，我们建议利用其他 listing 的现有的 embedding。在创建 listing 时，hosts 被要求提供关于 listing 的信息，例如位置、价格、listing 类型等。我们使用提供的关于 listing 的元数据来查找 $3$ 个地理上最接近的 listing (在10英里半径内)，它们具有 embedding 、与新的 listing 属于相同的 listing 类型(例如，包间)并且属于与新 listing 相同的价格区间(例如，$\$20−\$25$晚)。接下来，我们使用识别出的 listing 的 $3$ 个 embedding 来计算平均向量，以形成新的 listing embedding。使用这一技术，我们能够覆盖 $98\%$ 以上的新 listing。

#### Examining Listing Embeddings. 

To evaluate what characteristics of listings were captured by embeddings we examine the $d = 32$ dimensional embeddings trained using (5) on $800$ million click sessions. First, by performing k-means clustering on learned embeddings we evaluate if geographical similarity is encoded. Figure 2, which shows resulting $100$ clusters in California, confirms that listings from similar locations are clustered together. We found the clusters very useful for re-evaluating our definitions of travel markets. Next, we evaluate average cosine similarities between listings from Los Angeles of different listing types (Table 1) and between listings of different price ranges (Table 2). From those tables it can observed that cosine similarities between listings of same type and price ranges are much higher compared to similarities between listings of different types and price ranges. Therefore, we can conclude that those two listing characteristics are well encoded in the learned embeddings as well.

While some listing characteristics, such as price, do not need to be learned because they can be extracted from listing meta-data, other types of listing characteristics, such as architecture, style and feel are much harder to extract in form of listing features. 

To evaluate if these characteristics are captured by embeddings we can examine k-nearest neighbors of unique architecture listings in the listing embedding space. Figure 3 shows one such case where for a listing of unique architecture on the left, the most similar listings are of the same style and architecture. To be able to conduct fast and easy explorations in the listing embedding space we developed an internal Similarity Exploration Tool shown in Figure 4.

Demonstration video of this tool, which is available online at https://youtu.be/1kJSAG91TrI, shows many more examples of embeddings being able to find similar listings of the same unique architecture, including houseboats, treehouses, castles, chalets, beachfront apartments, etc.

> 为了评估 embedding 捕捉到 listing 的哪些特征，我们检查了在 $8$ 亿次点击session 上使用(5)训练的 $d=32$ 维 embedding。首先，通过对学习到的 embedding 进行 k-均值聚类，我们评估是否对地理相似性进行了编码。图2 显示了在加利福尼亚州产生的 100个簇，它确认了来自相似位置的 listing 是 聚类在一起的。我们发现这些簇对于重新评估我们对旅游市场的定义非常有用。接下来，我们评估来自洛杉矶的不同 listing type(表1)和不同价格范围(表2)的 listings 之间的平均余弦相似度。从这些表中可以观察到，与不同类型和价格范围的 listing 之间的相似性相比，相同类型和价格范围的 listing 之间的余弦相似性要高得多。因此，我们可以得出结论，这两个 listing 的特征在学习到的 embedding 中也得到了很好的编码。
>
> 虽然有些 listing 特征(如价格)不需要学习，因为它们可以从 listing 元数据中提取出来，但其他类型的 listing 特征，如建筑结构、风格和 feel，以 listing 特征的形式提取要困难得多。
>
> 为了评估这些特征是否被 embedding 捕获，我们可以在 listing embedding 空间中检查唯一的建筑结构的 listing 的 k 近邻。图3 显示了一个这样的例子，左边是一个独特的建筑结构 listing，最相似的 listing 具有相同的风格和建筑结构。为了能够在 listing embedding空间中进行快速而简单的探索，我们内部开发了一个相似性探索工具，如图4所示。
>
> 该工具的演示视频可在https://youtu.be/1kJSAG91TrI上在线获取，该视频显示了 embedding 的更多示例，它们能够找到相同独特建筑的相似 listing，包括船屋，树屋，城堡，小木屋，海滨公寓，等等。

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig1.png)

**Figure 1: Skip-gram model for Listing Embeddings**

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig2.png)

**Figure 2: California Listing Embedding Clusters**

![Figure3](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig3.png)

**Figure 3: Similar Listings using Embeddings**

![Figure4](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig4.png)

**Figure 4: Embeddings Evaluation Tool**

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table1.png)

**Table 1: Cosine similarities between different Listing Types**

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table2.png)

**Table 2: Cosine similarities between different Price Ranges**

#### 3.2 User-type & Listing-type Embeddings

Listing embeddings described in Section 3.1. that were trained using click sessions are very good at finding similarities between listings of the same market. As such, they are suitable for short-term, insession, personalization where the aim is to show to the user listings that are similar to the ones they clicked during the immanent search session.

However, in addition to in-session personalization, based on signals that just happened within the same session, it would be useful to personalize search based on signals from user’s longerterm history. For example, given a user who is currently searching for a listing in Los Angeles, and has made past bookings in New York and London, it would be useful to recommend listings that are similar to those previously booked ones.

While some cross-market similarities are captured in listing embeddings trained using clicks, a more principal way of learning such cross-market similarities would be to learn from sessions constructed of listings that a particular user booked over time. Specifically, let us assume we are given a set $\mathcal{S}_b$ of booking sessions obtained from $N$ users, where each booking session $s_b = (l_{b_1} , . . . ,l_{b_M})$ is defined as a sequence of listings booked by user $j$ ordered in time. Attempting to learn embeddings $v_{l_{id}}$ for each listing_id​ using this type of data would be challenging in many ways:

- First, booking sessions data $\mathcal{S}_b$ is much smaller than click sessions data $\mathcal{S}$ because bookings are less frequent events. 
- Second, many users booked only a single listing in the past and we cannot learn from a session of length $1$.
- Third, to learn a meaningful embedding for any entity from contextual information at least $5 − 10$ occurrences of that entity are needed in the data, and there are many listing_ids on the platform that were booked less than $5 − 10$ times.
- Finally, long time intervals may pass between two consecutive bookings by the user, and in that time user preferences, such as price point, may change, e.g. due to career change.

> 3.1节中描述的 listing embedding 是通过 click sessions 训练的，它非常善于发现同一市场的 listing 之间的相似之处。因此，它们适用于短期的 in-session 的个性化，其目的是向用户显示与他们在内部搜索 session 期间 click的 listing 相似的 listing。
>
> 然而，除了 in-session 的个性化之外，基于同一 session 内刚刚发生的信号进行个性化搜索，基于来自用户长期历史记录的信号进行个性化搜索也是有用的。例如，假设用户当前在洛杉矶搜索房源，并且过去曾在纽约和伦敦预订过房源，推荐与之前预订的房源相似的房源会很有用。
>
> 虽然在使用点击进行训练的 listing embedding 中捕获了一些跨市场的相似性，但是学习此类跨市场的相似性的更主要方法是从特定用户的随时间预订的 listing 中学习。 具体地，让我们假设给定了从 $N$ 个用户获得的一组预订 session $S_b$，其中每个预订 session $sb =(l_{b_1}，...，l_{b_M})$ 被定义为按时间排序的用户 $j$ 预订的一系列 listing。 尝试使用这种类型的数据来学习每个 listing_id 的 embedding $\textbf{v}_{l_{id}}$ 有许多挑战性：
>
> - 首先，预订 session 数据 $\mathcal{S}_b$ 要比单击 session 数据 $\mathcal{S}$ 小得多，因为预订是较不频繁的事件。
> - 其次，许多用户过去只预订了一份 listing ，我们无法从长度 $1$ 的 session 学到东西。
> - 第三，要想从上下文信息中了解到一个有意义的 embedding，至少需要在数据中出现 5 - 10 次，并且在平台上有许多 $listing\_id$ 被预订了 5 - 10 次。
> - 最后，用户连续两次预订之间可能会有很长的时间间隔，在这段时间内，用户的偏好(如价格点)可能会发生变化，例如由于职业变化。

To address these very common marketplace problems in practice, we propose to learn embeddings at a level of $listing\_type$ instead of $listing\_id$. Given meta-data available for a certain $listing\_id$ such as location, price, listing type, capacity, number of beds, etc., we use a rule-based mapping defined in Table 3 to determine its $listing\_type$. For example, an Entire Home listing from US that has a 2 person capacity, 1 bed, 1 bedroom & 1 bathroom, with Average Price Per Night of $\$60.8$, Average Price Per Night Per Guest of $\$29.3$, 5 reviews, all 5 stars, and 100% New Guest Accept Rate would map into 
$$
listing\_type={US}\_lt_1\_pn_3\_pg_3\_r_3\_5s_4\_c_2\_b_1\_bd_2\_bt_2\_nu_3
$$
Buckets are determined in a data-driven manner to maximize for coverage in each $listing\_type$ bucket. The mapping from $listing\_id$ to a $listing\_type$ is a many-to-one mapping, meaning that many listings will map into the same $listing\_type$.

To account for user ever-changing preferences over time we propose to learn user_type embeddings in the same vector space as listing_type embeddings. The user_type is determined using a similar procedure we applied to listings, i.e. by leveraging metadata about user and their previous bookings, defined in Table 4.

For example, for a user from San Francisco with MacBook laptop, English language settings, full profile with user photo, 83.4% average Guest 5 star rating from hosts, who has made 3 bookings in the past, where the average statistics of booked listings were \$52.52 Price Per Night, \$31.85 Price Per Night Per Guest, 2.33 Capacity, 8.24 Reviews and 76.1% Listing 5 star rating, the resulting user_type is 
$$
SF\_lg_1\_dt_1\_fp_1\_pp_1\_nb_1\_ppn_2\_ppg_3\_c_2\_nr_3\_l5s_3\_g5s_3.
$$
When generating booking sessions for training embeddings we calculate the user_type up to the latest booking. For users who made their first booking user_type is calculated based on the first 5 rows from Table 4 because at the time of booking we had no prior information about past bookings. This is convenient, because learned embeddings for user_types which are based on first 5 rows can be used for coldstart personalization for logged-out users and new users with no past bookings.

> 为了在实践中解决这些非常常见的市场问题，我们建议在 $listing\_type$ 级别学习嵌入式，而不是 $listing\_id$级别。给定某个 $listing\_id$可用的元数据，例如 位置、价格、listing type、容量、床数等，我们使用 表3 中定义的基于规则的映射来确定它的 $listing\_type$。例如，一个完整的房源 listing ，从我们的2人，1床，1卧室和1浴室，平均每晚价格为 $\$60.8$，平均每晚每位客人的价格为 $\$29.3$，5个评论，所有的 5 星级，和 100% 的新客人接受率将映射到
>
> $$
> listing\_type ={US}\_lt_1\_pn_3\_pg_3\_r_3\_5s_4\_c_2\_b_1\_bd_2\_bt_2\_nu_3
> $$
> 桶以数据驱动的方式确定，以最大限度地覆盖每个 $listing\_type$存储桶。从 $listing\_id$ 到 $listing\_type$ 的映射是一个多对一的映射，这意味着许多listing 将映射到相同的 $listing\_type$。
>
> 例如，对于来自旧金山且拥有 MacBook 笔记本电脑的用户，英语环境，个人资料完整（带有照片），来自房东的平均 5 星宾客评级，过去进行了 3 次预订，预订的统计为平均 \$52.52 每晚，\$31.85每人，2.33人，8.24评论和 76.1% listing 评价 5 星，得到的 user_type 为
> $$
> SF\_lg_1\_dt_1\_fp_1\_pp_1\_nb_1\_ppn_2\_ppg_3\_c_2\_nr_3\_l5s_3\_g5s_3.
> $$
> 在为训练 embedding 生成预订 session 时，我们计算最新预订的 user_type。对于第一次预订的用户，USER_TYPE是根据表 4 中的前 5 行计算的，因为在预订时，我们没有关于过去预订的信息。这很方便，因为基于前5行的 $user\_type$ 的学习 embedding 可以用于未登录用户和过去没有预订的新用户的冷启动个性化。

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table3.png)

**Table 3: Mappings of listing meta data to listing type buckets**

![Table4](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table4.png)

**Table 4: Mappings of user meta data to user type buckets**

#### Training Procedure. 

To learn $user\_type$ and $listing\_type$ embeddings in the same vector space we incorporate the $user\_type$ into the booking sessions. Specifically, we form a set $\mathcal{S}_b$ consisting of $N_b$ booking sessions from $N$ users, where each session $s_b = (u_{type_1},l_{type_1}, \cdots ,u_{type_M},l_{type_M}) \in \mathcal{S}_b$ is defined as a sequence of booking events, i.e. ($user\_type$ , $listing\_type$) tuples ordered in time. Note that each session consists of bookings by same $user\_id$, however for a single $user\_id$ their $user\_types$ can change over time, similarly to how $listing\_types$ for the same listing can change over time as they receive more bookings.

The objective that needs to be optimized can be defined similarly to (3), where instead of listing $l$ , the center item that needs to be updated is either $user\_type$ ($u_t$) or $listing\_type$ ($l_t$) depending on which one is caught in the sliding window. For example, to update the central item which is a $user\_type$ ($u_t$) we use
$$
\mathop{argmax}_{\theta}
\sum_{(u_t,c) \in \mathcal{D}_{book}}
log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_{u_t}}}
+
\sum_{(u_t,c)\in\mathcal{D}_{neg}}
log\frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_{u_t}}},

\qquad(6)
$$
where $\mathcal{D}_{book}$ contains the $user\_type$ and $listing\_type$ from recent user history, specifically user bookings from near past and near future with respect to central item’s timestamp, while $\mathcal{D}_{neg}$ contains random $user\_type$ or $listing\_type$ instances used as negatives. Similarly, if the central item is a $listing\_type$ ($l_t$) we optimize the following objective
$$
\mathop{argmax}_{\theta}
\sum_{(l_t,c) \in \mathcal{D}_{book}}
log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_{l_t}}}
+
\sum_{(l_t,c) \in \mathcal{D}_{neg}}
log \frac{1}{1+e^{\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}

\qquad (7)
$$
Figure 5a (on the left) shows a graphical representation of this model, where central item represents $user\_type$ ($u_t$) for which the updates are performed as in (6).

Since booking sessions by definition mostly contain listings from different markets, there is no need to sample additional negatives from same market as the booked listing, like we did in Session 3.1. to account for the congregated search in click sessions.

> 为了在同一向量空间中学习 $user\_type$ 和 $listing\_type$ embedding，我们将$user\_type$ 合并到预订 session 中。 具体来说，我们形成一个集合 $\mathcal{S}_b$，由来自 $N$ 个用户的 $N_b$个预订 session 组成，其中每个 session $s_b = (u_{type_1},l_{type_1}, \cdots ,u_{type_M},l_{type_M}) \in \mathcal{S}_b$ 被定义为预订事件的序列，即以时间排序的 ($user\_type$，$listing\_type$) 元组。 请注意，每个 session 都由相同的 $user\_id$ 进行预订，但是对于单个 $user\_id$，其 $user\_type$ 可以随时间变化而变化，这与相同 listing 的 $listing\_types$ 随着接收更多预订而可以随时间变化的方式类似。
>
> 需要优化的目标的定义类似于 (3)，其中需要更新的中心项不是 listing $l$，而是 $user\_type$ ($u_t$) 或 $listing\_type$ ($l_t$)，这取决于在滑动窗口中捕获的是哪一个。例如，要更新我们使用的 $user\_type$ ($u_t$) 的中心项
> $$
> \mathop{argmax}_{\theta}
> \sum_{(u_t,c) \in \mathcal{D}_{book}}
> log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_{u_t}}}
> +
> \sum_{(u_t,c)\in\mathcal{D}_{neg}}
> log\frac{1}{1+e^{\textbf{v}_c^{'}\textbf{v}_{u_t}}},
> 
> \qquad(6)
> $$
> 其中，$\mathcal{D}_{book}$ 包含最近用户历史记录中的 $user\_type$ 和 $listing\_type$，特别是相对于中心项目的时间戳来自过去和未来的用户预订，而 $\mathcal{D}_{neg}$ 包含用作负样本的随机 $user\_type$ 或 $listing\_type$ 实例。
> 类似地，如果中心项目是 listing 类型($l_t$)，我们将优化以下目标
> $$
> \mathop{argmax}_{\theta}
> \sum_{(l_t,c) \in \mathcal{D}_{book}}
> log \frac{1}{1+e^{-\textbf{v}_c^{'}\textbf{v}_{l_t}}}
> +
> \sum_{(l_t,c) \in \mathcal{D}_{neg}}
> log \frac{1}{1+e^{\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}
> 
> \qquad (7)
> $$
> Figure 5a(左侧)显示了该模型，其中中心项表示 $user\_type$($u_t$)，对其更新，如(6)所示。
>
> 由于预订 sessions 通常包含来自不同市场的 listing，所以没有必要像我们在3.1 中所做的那样，为了优化在 click session 内的聚合搜索，从同一市场的 listing 中采样额外的负样本。

![Figure5](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Fig5.png)

**Figure 5: Listing Type and User Type Skip-gram model**

#### Explicit Negatives for Rejections.

Unlike clicks that only reflect guest-side preferences, bookings reflect host-side preferences as well, as there exists an explicit feedback from the host, in form of accepting guest’s request to book or rejecting guest’s request to book. Some of the reasons for host rejections are bad guest star ratings, incomplete or empty guest profile, no profile picture, etc. These characteristics are part of user\_type definition from Table 4.

Host rejections can be utilized during training to encode the host preference signal in the embedding space in addition to the guest preference signal. The whole purpose of incorporating the rejection signal is that some $listing\_types$ are less sensitive to $user\_types$ with no bookings, incomplete profiles and less than average guest star ratings than others, and we want the embeddings of those $listing\_types$ and $user\_types$ to be closer in the vector space, such that recommendations based on embedding similarities would reduce future rejections in addition to maximizing booking chances.

We formulate the use of the rejections as explicit negatives in the following manner. In addition to sets $\mathcal{D}_{book}$ and $ \mathcal{D}_{neg}$, we generate a set $\mathcal{D}_{rej}$ of pairs ($u_t$ , $l_t$ ) of $user\_type$ or $listing\_type$ that were involved in a rejection event. As depicted in Figure 5b (on the right), we specifically focus on the cases when host rejections (labeled with a minus sign) were followed by a successful booking (labeled with a plus sign) of another listing by the same user. The new optimization objective can then be formulated as
$$
\mathop{argmax}_{\theta}
\sum_{(u_t,c) \in \mathcal{D}_{book}} 
log \frac{1}{1+exp^{-\textbf{v}_{c}^{'}\textbf{v}_{u_t}}}
+
\sum_{(u_t,c) \in \mathcal{D}_{neg}}
log\frac{1}{1+exp^{\textbf{v}_c^{'}\textbf{v}_{u_t}}}
\\
+
\sum_{u_t,l_t \in \mathcal{D}_{reject}}
log \frac{1}{1+exp^{\textbf{v}_{l_t}^{'}\textbf{v}_{u_t}}}
\qquad (8)
$$
in case of updating the central item which is a $user\_type$ ($u_t$)。
$$
\mathop{argmax}_{\theta}
\sum_{(l_t,c) \in \mathcal{D}_{book}}
log \frac{1}{1+exp^{-\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}
+
\sum_{(l_t,c) \in \mathcal{D}_{neg}}
log \frac{1}{1+exp^{\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}
\\
+
\sum_{(l_t,u_t) \in \mathcal{D}_{reject}} log \frac{1}{1+exp^{\textbf{v}_{u_t}^{'}\textbf{v}_{l_t}}}

\qquad(9)
$$
in case of updating the central item which is a $listing\_type$ ($l_t$)。

Given learned embeddings for all $user\_types$ and $listing\_types$, we can recommend to the user the most relevant listings based on the cosine similarities between user’s current $user\_type$ embedding and $listing\_type$ embeddings of candidate listings.

For example, in Table 5 we show cosine similarities between 
$$
user\_type = {SF}\_{lg_1}\_dt_1\_fp_1\_pp_1\_nb_3\_ppn_5\_ppg_5\_c_4\_nr_3\_l5s_3\_g5s_3
$$
who typically books high quality, spacious listings with lots of good reviews and several different $listing\_type$s in US. It can be observed that listing types that best match these user preferences, i.e. entire home, lots of good reviews, large and above average price, have high cosine similarity, while the ones that do not match user preferences, i.e. ones with less space, lower price and small number of reviews have low cosine similarity.

> 与只反映 guest 的偏好的点击不同，预订也反映了 host 的偏好，因为有来自 host 以接受 guest 的预订请求或拒绝 guest 的预订请求的形式的明确反馈。host 拒绝的一些原因是 guest 星级评分差、guest 资料不完整或空、没有资料图片等。这些特性是表 4 中 $user\_type$ 定义的一部分。
>
> 在训练期间，除了 guest 偏好信号之外，还可以利用 host 拒绝来编码 embedding 空间中的 host 偏好信号。加入拒绝信号的整个目的是，与其他类型相比，一些 listing type对没有预订、profile 不完整、guest 星级评分低于平均水平的 $user\_type$ 不那么敏感，我们希望这些 listing\_type 和 user\_type 的 embedding 在向量空间中更紧密，这样基于 embedding 相似性的推荐将在最大化预订机会的同时减少未来的拒绝。
>
> rejections 的用法是显式的负样本。除了集合 $\mathcal{D}_{book}$ 和 $\mathcal{D}_{neg}$ 之外，我们还生成了一个集合 $\mathcal{D}_{rej}$， 包含 ($u_t$,$l_t$) 是拒绝事件的 $user\_type$ 或 $listing\_type$ 对。
>
> 如图5b(右)所示，我们特别关注这样一种情况:同一用户在 host 拒绝(标记为减号)之后成功预订了另一个 listing (标记为加号)。
>
> 新的优化目标可以表述为
> $$
> \mathop{argmax}_{\theta}
> \sum_{(u_t,c) \in \mathcal{D}_{book}} 
> log \frac{1}{1+exp^{-\textbf{v}_{c}^{'}\textbf{v}_{u_t}}}
> +
> \sum_{(u_t,c) \in \mathcal{D}_{neg}}
> log\frac{1}{1+exp^{\textbf{v}_c^{'}\textbf{v}_{u_t}}}
> \\
> +
> \sum_{u_t,l_t \in \mathcal{D}_{reject}}
> log \frac{1}{1+exp^{\textbf{v}_{l_t}^{'}\textbf{v}_{u_t}}}
> \qquad (8)
> $$
> (8) 为 $user\_type$ ($u_t$) 的更新。
> $$
> \mathop{argmax}_{\theta}
> \sum_{(l_t,c) \in \mathcal{D}_{book}}
> log \frac{1}{1+exp^{-\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}
> +
> \sum_{(l_t,c) \in \mathcal{D}_{neg}}
> log \frac{1}{1+exp^{\textbf{v}_{c}^{'}\textbf{v}_{l_t}}}
> \\
> +
> \sum_{(l_t,u_t) \in \mathcal{D}_{reject}} log \frac{1}{1+exp^{\textbf{v}_{u_t}^{'}\textbf{v}_{l_t}}}
> 
> \qquad(9)
> $$
> (9) 为 $listing\_type$ 的更新。
>
> 已知所有 $user\_types$ 和 $listing\_types$，我们可以根据用户当前的 $user\_type$ embedding 和候选 $listing$ 的 $listing\_type$ embedding 之间的余弦相似性，向用户推荐最相关的 $listing$。
>
> 例如，在 Table 5中，我们展示了以下之间的余弦相似性:
> $$
> user\_type = {SF}\_{lg_1}\_dt_1\_fp_1\_pp_1\_nb_3\_ppn_5\_ppg_5\_c_4\_nr_3\_l5s_3\_g5s_3
> $$
> 他们通常在美国预订高质量、宽敞的房源，有很多好评和几种不同的 listing\_type。可以观察到，与这些用户偏好最匹配的 $listing\_type$，即整租、大量好评、大，且高于平均价格的 $listing\_type$ 具有较高的余弦相似度，而与用户偏好不匹配的 $listing\_type$，即具有较小空间、较低价格和较少评论的 $listing\_type$具有较低的余弦相似度。

![Table5](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Real-time Personalization using Embeddings for Search Ranking at Airbnb/Table5.png)

**Table 5: Recommendations based on type embeddings**