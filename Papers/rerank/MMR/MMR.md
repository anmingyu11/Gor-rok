# The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

## **Abstract**

This paper presents a method for combining query-relevance with information-novelty in the context of text retrieval and summarization. The Maximal Marginal Relevance (MMR) criterion strives to reduce redundancy while maintaining query relevance in re-ranking retrieved documents and in selecting apprw priate passages for text summarization. Preliminary results indicate some benefits for MMR diversity ranking in document retrieval and in single document summarization. The latter are borne out by the recent results of the SUMMAC conference in the evaluation of summarization systems. However, the clearest advantage is demonstrated in constructing non-redundant multi-document summaries, where MMR results are clearly superior to non-MMR passage selection. 

> 本文提出了一种在 text retrieval 和 summarization 中将查询相关性和信息新颖性相结合的方法。最大边际相关性(maximum Marginal Relevance, MMR)准则在对检索文档进行重排和为文本摘要选择合适的段落时，力求在保持查询相关性的同时减少冗余。初步结果表明，MMR多样性排序在文档检索和单文档摘要中具有一定的优势。后者在 SUMMAC 会议最近评价摘要系统的结果中得到证实。然而，最明显的优势是构建无冗余的多文档摘要，其中MMR结果明显优于非MMR段落选择。

### 1. Introduction

With the continuing growth of online information, it has become increasingly important to provide improved mechanisms to find information quickly. Conventional IR systems rank and assimilate documents based on maximizing relevance to the user query [1, 5]. In cases where relevant documents are few, or cases where very-high recall is necessary, pure relevance ranking is very appropriate. But in cases where there is a vast sea of potentially relevant documents, highly redundant with each other or (in the extreme) containing partially or fully duplicative information we must utilize means beyond pure relevance for document ranking. 

A new document ranking method is one where each document in the ranked list is selected according to a combined criterion of query relevance and novelty of information. The latter measures the degree of dissimilarity between the document being considered and previously selected ones already in the ranked list. Of course,some users may prefer to drill down on a narrow topic, and others a panoramic sampling bearing relevance to the query. Best is a user-tunable method; Maximal Marginal Relevance (MMR) provides precisely such functionality,as discussed below. 

> 随着线上信息的不断增长，提供更完善的机制来快速查找信息变得越来越重要。传统的IR系统基于与用户查询的最大相关性对文档进行排序和吸收(assimilate)[l, 51]。在相关文档很少的情况下，或者需要很高的召回率的情况下，纯相关性排名是非常合适的。但是，如果有大量潜在相关的文档，这些文档彼此之间高度冗余，或者(在极端情况下)包含部分或全部重复的信息，我们必须使用纯相关性以外的方法进行文档排序。
>
> 一种新的文档排序方法是根据查询的相关性和信息的新颖性相结合的标准来选择排序列表中的每个文档。后者衡量的是正在审议的文件与已列入排序的文件之间的差异程度。当然，有些用户可能更喜欢钻研一个狭窄的主题，而另一些用户可能更喜欢查询相关的全景采样。Best 是一种用户可调的方法； 如下所述，最大边际相关性 (MMR) 恰好提供了此类功能。

## 2. Maximal Marginal Relevance 

Most modem IR search engines produce a ranked list of retrieved documents ordered by declining relevance to the user’s query. In contrast, we motivated the need for “relevant novelty” as a potentially superior criterion. A first approximation to measuring relevant novelty is to measure relevance and novelty independently and provide a linear combination as the metric. We call the linear combination “marginal relevance” - i.e. a document has high marginal relevance if it is both relevant to the query and contains minimal similarity to previously selected documents. We strive to maximize-marginal relevance in retrieval and summarization, hence we label our method “maximal marginal relevanci” (MMR).
$$
M M R \stackrel{\text { def }}{=} \operatorname{arg} \max _{D_{i} \in R \backslash S}\left[\lambda\left(\operatorname{Sim}_{1}\left(D_{i}, Q\right)-(1-\lambda) \max _{D_{j} \in S} \operatorname{Sim}_{2}\left(D_{i}, D_{j}\right)\right)\right]
$$
Where $C$​​​​ is a document collection (or document stream); $Q$​​​ is a query or user profile; $R = IR(C, Q, \theta)$​​​, i.e., the ranked list of documents retrieved by an IR system, given $C$​​​ and $Q$​​​ and a relevance threshold $8$​​​, below which it will not retrieve documents ($\theta$​​​ can be degree of match or number of documents); $S$​​​ is the subset of documents in $R$​​​ already selected; $R \backslash S$​​​ is the set difference, i.e, the set of as yet unselected documents in $R$​​​; Siml is the similarity metric used in document retrieval and relevance ranking between documents (passages) and a query; and Sim2 can be the same as Sim1 or a different metric. 

> 大多数现代IR搜索引擎生成一个检索文档的排序列表，该列表根据与用户查询的相关度按递减进行排序。与之相反，我们将“新颖的相关性”作为潜在的优先标准来激发需求。评估“新颖的相关性”的第一个近似方法是独立评估相关性和新颖性，并提供一个线性组合作为度量。我们称这种线性组合为“边际相关性”——即，如果一个文档既与查询相关，又与之前选择的文档具有最小的相似性，那么该文档就具有较高的边际相关性。我们努力在检索和摘要中最大化边际相关性，因此我们将我们的方法标记为“最大边际相关性”(MMR)。
> $$
> M M R \stackrel{\text { def }}{=} \operatorname{arg} \max _{D_{i} \in R \backslash S}\left[\lambda\left(\operatorname{Sim}_{1}\left(D_{i}, Q\right)-(1-\lambda) \max _{D_{j} \in S} \operatorname{Sim}_{2}\left(D_{i}, D_{j}\right)\right)\right]
> $$
> $C$ 是一个文档集合(或文档流);
>
> $Q$ 是一个查询或用户简介;
>
> $R = IR(C, Q, \theta)$​，即 IR 系统检索的文档排序列表，给定 $C$​、$Q$​ 和相关性阈值 $8$​，低于该阈值的文档不会被检索出($\theta$​​ 可以是匹配程度或文档数量);
>
> $S$ 是 $R$ 中已经选择的文档子集;
>
> $R \backslash S$ 是集合差，即 $R$ 中尚未选定的文档的集合;
>
> Sim1 是用于文档检索和文档(段落)与查询之间关联排序的相似度度量;
>
> Sim2 可以和 Sim1 相同，也可以是不同的度量。

Given the above definition, MMR computes incrementally the standard relevance-ranked list when the parameter $\lambda =1$, and computes a maximal diversity ranking among the documents in $R$ when $\lambda=0$. For intermediate values of $\lambda$ in the interval $[0,1]$, a linear combination of both criteria is optimized. Users wishing to sample the information space around the query, should set $\lambda$ at a smaller value, and those wishing to focus in on multiple potentially overlapping or reinforcing relevant documents, should set $\lambda$​ to a value closer to $\lambda$​. We found that a particularly effective search strategy (reinforced by the user study discussed below) is to start with a small $\lambda$ (e.g. $\lambda = .3$) in order to understand the information space in the region of the query, and then to focus on the most important parts using a reformulated query (possibly via relevance feedback) and a larger value of $\lambda$ (e.g. $\lambda = .7$). 

> 鉴于上述定义，当参数 $\lambda=1$ 时，MMR 递增计算标准的相关性排序列表，并在 $\lambda=0$ 时计算 $R$​​ 中文档之间的最大多样性排名。
>
> 对于区间 $[0,1]$ 中 $\lambda$ 的中间值，优化了两个标准的线性组合。
>
> 希望对查询周围的信息空间进行采样的用户应该将 $\lambda$​ 设置为较小的值，而那些希望关注多个可能重叠或增强的相关文档的用户应该将 $\lambda$​ 设置为更接近 $\lambda$​ 的值(？？？？)。
>
> 我们发现一个特别有效的搜索策略（由下面讨论的用户研究加强）是从一个小的 $\lambda$（例如 $ \lambda= .3$）开始，以了解查询区域中的信息空间，然后专注于最重要的部分使用重新制定的查询（可能通过相关性反馈）和更大的 $\lambda$ 值（例如 $ \lambda = .7$）。

## 3 Document Reordering

We performed a pilot experiment with five users who were undergraduates from various disciplines. The purpose of the study was to find out if they could tell what was the difference between a standard ranking method and MMR. The users were asked to find information from documents and were not told how the order in which documents were presented - only that either “method R” or “method S” were used. The majority of people said they preferred the method which gave in their opinion the most broad and interesting topics (MMR). In the final section they were asked to select a search method and use it for a search task. 80% chose the method MMR. The users indicated a differential preference for MMR in navigation and for locating the relevant candidate documents more quickly, and pure-relevance ranking when looking at related documents within that band. Three of the five users clearly discovered the differential utility of diversity search and relevance-only search.

> 我们对五名来自不同学科的本科生进行了试点实验。
>
> 该研究的目的是找出他们是否能说出标准排序方法和 MMR 之间的区别。
>
> 用户被要求从文档中查找信息，而没有被告知文档的呈现顺序——只使用了 “方法 R” 或 “方法 S” 。 大多数人表示他们更喜欢在他们看来给出最广泛和最有趣的话题 (MMR) 的方法。
>
> 在最后一部分，他们被要求选择一种搜索方法并将其用于搜索任务。 80% 的人选择了 MMR 方法。
>
> 用户表示了在导航和更快地定位相关候选文档时对 MMR 的不同偏好，以及在查看该频段内的相关文档时的纯相关性排名。 五个用户中的三个清楚地发现了多样性搜索和仅相关搜索的不同效用。

## 4 Summarization 

If we consider document summarization by relevantpassage extraction, we must again consider relevance as well as anti-redundancy. Summaries need to avoid redundancy, as it defeats the purpose of summarization. If we move beyond single document summarization to document cluster summarization, where the summary must pool passages from different but possibly overlap ping documents, reducing redundancy becomes an even more significant problem.

Automated document summarization dates back to Luhn’s work at IBM in the 1950’s [4], and evolved through several efforts to the recent TIPSTER effort which includes trainable methods [3], linguistic approaches [S] and our information-centric method [2], the first to focus on anti-redundancy measures.

Human summarization of documents, sometimes called “abstraction” is a fixed-length summary, reflecting the key points that the abstractor - rather than the user -deems important. A different user with different information needs may require a totally different summary of the same document. We created single document summaries by segmenting the document into passages (sentences in our case) and using MMR with a cosine similarity metric to rerank the passages in response to a user generated or system generated query. The top ranking passages were presented in the original document order. 

In the May 1998 SUMMAC conference [6], featuring a government-run evaluation of 15 summarization systems, our MMR-based summarizer produced the highestutility query-relevant summaries with an F-score of .73 derived from precision and recall by assessors making topic-relevance judgements from summaries. Our system also scored highest (70% accuracy) on informative summaries, where the assessor judged whether the summary contained the information required to answer a set of key questions. It should be noted that some parameters, such as summary length, varied among systems and therefore the evaluation results are indicative but not definitive measures of comparative performance. 

In order to evaluate what the relevance loss for a diversity gain in single document summarization, three assessors went through 50 articles from 200 articles of a TIPSTER topic and marked each sentence as relevant, somewhat relevant and irrelevant. The article was also marked as relevant or irrelevant. The assessor scores were compared against the TREC relevance judgments provided for the topic. 

The sentence precision results are given in Table 1 for compression factors .25 and .1. Two precision scores were calculated, (1) that of TREC relevance plus at least one CMU assessor marking the document as relevant (yielding 23 documents) and (2) at least two of the three CMU assessors marking the document as relevant (yielding 18 documents). From these scores we can see there is no significant statistical difference between the $\lambda=1$​, $\lambda=.7$​, and $\lambda=.3$​ scores. This is often explained by cases where the $\lambda=1$​ summary failed to pick up a piece of relevant information and the reranking with $\lambda=.7$​ or $.3$​ might.

The MMR-passage selection method for summarization works better for longer documents (which typically contain more inherent passage redundancy across document sections such as abstract, introduction, conclusion, results, etc.). MMR is also extremely useful in extraction of passages from multiple documents about the same topics. News stories contain much repetition of background information. Our preliminary results for multi-document summarization show that in the top 10 passages returned for news story collections in response to a query, there is significant repetition in content over the retrieved passages and the passages often contain duplicate or nearreplication in the sentences. MMR reduces or eliminates such redundancy. 

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/rerank/MMR/Fig1.png)

**Table1. Precision Scores**

> 如果我们通过相关性和段落提取来考虑文档摘要，我们必须再次考虑相关性和抗冗余性。摘要需要避免冗余，因为它违背了摘要的目的。如果我们从单个文档摘要转移到文档集群摘要，其中摘要必须汇集来自不同但可能重叠的文档的段落，那么减少冗余就成为一个更重要的问题。
>
> 自动文档摘要可以追溯到 1950 年代 Luhn 在 IBM 的工作 [4]，并通过多次努力演变为最近的 TIPSTER ，其中包括可训练方法 [3]、语言方法 [S] 和我们以信息为中心的方法 [2]， 首先要注重反冗余措施。
>
> 文档的人工摘要(有时称为“摘要”)是固定长度的摘要，反映了抽象者(而不是用户)认为重要的关键点。具有不同信息需求的不同用户可能需要相同文档的完全不同的摘要。我们通过将文档分割成段落(在我们的例子中是句子)并使用带有余弦相似性度量的MMR来响应用户生成或系统生成的查询来重新排序段落，从而创建了单个文档摘要。排名靠前的段落按原始文档顺序排列。
>
> 在1998年5月举行的SUMMAC会议[6]上，政府对15个摘要系统进行了评估，我们基于 mmr 的摘要产生了效用最高的查询相关摘要，由评估人员从摘要中做出主题相关判断的准确性和召回率得出F-score为0.73。我们的系统在信息性摘要上也得分最高（70% 的准确率），其中评估员判断摘要是否包含回答一组关键问题所需的信息。 应该注意的是，一些参数（例如摘要长度）因系统而异，因此评估结果是指示性的，但不是比较性能的确定性度量。
>
> 为了评估在单个文档摘要中，相关性损失对多样性的增加有什么影响，三位评审员从一个 TIPSTER 的 200 篇文章中阅读了 50 篇文章，并将每句话标记为相关、有点相关和不相关。这篇文章也被标记为相关或无关。评审员的分数与TREC为该主题提供的相关性判断进行了比较。
>
> 表1给出了压缩系数为.25和.1的语句精度结果。计算了两个精确度分数，(1)TREC相关性评分加上至少一名CMU评估员将该文件标记为相关(产生23份文件)和(2)三名CMU评估员中至少两名将该文件标记为相关(产生18份文件)。从这些分数可以看出，\lambda=1、\lambda=0.7和\lambda=0.3分数之间没有显著的统计学差异。这通常可以通过以下情况来解释：\lambda=1摘要无法提取一条相关信息，而使用\lambda=.7或.3进行重新排序可能会造成这种情况。
>
> 表1给出了压缩系数为.25和.1的语句的精准率。计算了两个精确度分数，(1)TREC相关性评分加上至少一名 CMU 评估员将该文件标记为相关(产生23份文件)和(2)三名 CMU 评估员中至少两名将该文件标记为相关(产生18份文件)。从这些分数可以看出，$\lambda=1$​、$\lambda=0.7$ ​和 $\lambda=0.3$​ 分数之间没有显著的统计学差异。这通常可以通过以下情况来解释：$\lambda=1$ 摘要无法提取一条相关信息，而使用 $\lambda=.7$ 或 $.3$​​ 进行重排能。
>
> 用于摘要的 MMR-passage 选择方法对于较长的文档更有效(通常在文档各部分中包含更多固有的段落冗余，如摘要、引言、结论、结果等)。MMR在从关于相同主题的多个文档中提取段落时也非常有用。新闻报道包含许多重复的背景信息。我们对多文档摘要的初步结果显示，在响应查询的新闻故事集返回的前10个段落中，检索到的段落在内容上有显著的重复，段落在句子中经常包含重复或接近重复的内容。MMR减少或消除了这种冗余。



## 5 Concluding Remarks 

We have shown that MMR ranking provides a useful and beneficial manner of providing information to the user by allowing the user to minimize redundancy. This is especially true in the case of query-relevant multi-document summarization. We are currently performing studies on how this extends to several document collections as well as studies on the effectiveness of our system. 

> 我们已经证明，MMR 排序通过允许用户最小化冗余，为用户提供了一种有用和有益的信息方式。在与查询相关的多文档摘要的情况下尤其如此。我们目前正在进行研究，研究这如何扩展到几个文档集合，以及研究我们的系统的有效性。
