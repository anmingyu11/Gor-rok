## 5. EXPERIMENTAL DESIGN

In this section we provide an overview of the datasets and methods which we will use in our experiments. Code and data to reproduce our results will be available at the first author’s website.1

> 在本节中，我们概述了我们将在实验中使用的数据集和方法。复制我们结果的代码和数据将在第一作者的网站上提供。

## 5.1 Datasets

An overview of the graphs we consider in our experiments is given in Figure 1. 

- BlogCatalog [39] is a network of social relationships provided by blogger authors. The labels represent the topic categories provided by the authors.
- Flickr [39] is a network of the contacts between users of the photo sharing website. The labels represent the interest groups of the users such as ‘black and white photos’.
- YouTube [40] is a social network between users of the popular video sharing website. The labels here represent groups of viewers that enjoy common video genres (e.g. anime and wrestling).

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Table1.png)

**Table 1: Graphs used in our experiments.**

> 图1 给出了我们在实验中考虑的图的概述。
>
> - BlogCatalog[39]是一个由博客作者提供的社会关系网络。标签代表作者提供的主题类别。
> - Flickr[39]是照片共享网站用户之间的联系人网络。标签代表了用户的兴趣组，比如“黑白照片”。
> - YouTube[40]是流行的视频分享网站用户之间的社交网络。这里的标签代表了喜欢常见视频类型(例如 动漫 和 摔跤 )的观众群体。

## 5.2 Baseline Methods

To validate the performance of our approach we compare it against a number of baselines: 

- SpectralClustering [41]: This method generates a representation in $\mathbb{R}^d$ from the $d$-smallest eigenvectors of $\widetilde{L}$, the normalized graph Laplacian of $G$ . Utilizing the eigenvectors of $\widetilde{L}$ implicitly assumes that graph cuts will be useful for classification.
- Modularity [39]: This method generates a representation in $\mathbb{R}^d$ from the top-$d$ eigenvectors of $B$, the Modularity matrix of $G$. The eigenvectors of $B$ encode information about modular graph partitions of $G$ [35]. Using them as features assumes that modular graph partitions will be useful for classification.
- EdgeCluster [40]: This method uses k-means clustering to cluster the adjacency matrix of $G$. Its has been shown to perform comparably to the Modularity method, with the added advantage of scaling to graphs which are too large for spectral decomposition.
- wvRN [25]: The weighted-vote Relational Neighbor is a relational classifier. Given the neighborhood $\mathcal{N}_i$ of vertex $v_i$, wvRN estimates $Pr(y_i|\mathcal{N}_i)$ with the (appropriately normalized) weighted mean of its neighbors (i.e $Pr(y_i|\mathcal{N}_i) = \frac{1}{Z} \sum_{v_j \in \mathcal{N}_i}  \ w_{ij} \ Pr(y_j | \mathcal{N}_j )$). It has shown surprisingly good performance in real networks, and has been advocated as a sensible relational classification baseline [26].
- Majority: This naive method simply chooses the most frequent labels in the training set.

> - SpectralClustering [41]：此方法从 $\widetilde{L}$ 的 $d$ 最小特征向量( $G$ 的归一化图Laplacian)生成 $\mathbb{R}^d$ 中的表示。利用 $\widetilde{L}$ 的特征向量假设图 cuts对分类有用。
> - Modularity [39]：此方法从 $B$ 的 top-$d$ 特征向量( $G$ 的 Modularity 矩阵)生成 $\mathbb{R}^d$ 的表示。$B$ 的特征向量编码了有关 $G$ [35]的 modular 图的分区信息。使用它们作为特征假设 modular 图分区将对分类有用。
> - EdgeCluster[40]：这种方法使用 k-means 聚类来对 $G$ 的邻接矩阵进行聚类。已经证明，这种方法的性能与 Modularity 方法相当，而且还可以扩展到对于谱聚类来说太大的图。
> - wvRN[25]：weighted-vote relational neighbor 是关系分类器。给定顶点 $v_i$ 的邻域 $\mathcal{N}_i$，wvRN用其邻域的(适当归一化)加权平均值来估计$Pr(y_i|N_i)$ (即 $Pr(y_i|\mathcal{N}_i) = \frac{1}{Z} \sum_{v_j \in \mathcal{N}_i}  \ w_{ij} \ Pr(y_j | \mathcal{N}_j )$)。它在真实 networks 中表现得出人意料地好，并被认为是一种合理的关系分类基线[26]。
> - Majority：这种简单的方法只是在训练集中选择最频繁的标签。

## 6. EXPERIMENTS

In this section we present an experimental analysis of our method. We thoroughly evaluate it on a number of multilabel classification tasks, and analyze its sensitivity across several parameters.

> 在本节中，我们将对我们的方法进行实验分析。我们在多个多标签分类任务上对其进行了全面的评估，并分析了它对几个参数的敏感性。

## 6.1 Multi-Label Classification

To facilitate the comparison between our method and the relevant baselines, we use the exact same datasets and experimental procedure as in [39, 40]. Specifically, we randomly sample a portion ($T_R$) of the labeled nodes, and use them as training data. The rest of the nodes are used as test. We repeat this process 10 times, and report the average performance in terms of both Macro-F1 and Micro-F1. When possible we report the original results [39, 40] here directly.

For all models we use a one-vs-rest logistic regression implemented by LibLinear [11] extended to return the most probable labels as in [39]. We present results for DeepWalk with ($\gamma = 80$，$w = 10$，$d = 128$). The results for (SpectralClustering, Modularity, EdgeCluster) use Tang and Liu’s preferred dimensionality, $d = 500$.

> 为了便于我们的方法与相关 baseline 之间的比较，我们使用了与[39，40]中完全相同的数据集和实验程序。具体地说，我们随机抽样已标记节点的一部分($T_R$)，并将它们用作训练数据。其余节点用作测试。我们重复此过程10次，并根据 Macro-F1 和 Micro-F1 报告平均性能。如果实验可能进行，我们直接在这里报告原始结果[39，40]。
>
> 对于所有模型，我们都使用由 Libline[11]扩展实现的 one-vs-rest逻辑回归，以返回最可能的标签，如[39]所示。我们给出了 DeepWalk ($\gamma=80$，$w=10$，$d=128$) 的结果。(SpectralCluging，Modulality，EdgeCluster)的结果使用了Tang 和 Liu 的偏好维数，$d=500$。

#### 6.1.1 BlogCatalog

In this experiment we increase the training ratio ($T_R$) on the BlogCatalog network from 10% to 90%. Our results are presented in Table 2. Numbers in bold represent the highest performance in each column.

DeepWalk performs consistently better than EdgeCluster, Modularity, and wvRN. In fact, when trained with only 20% of the nodes labeled, DeepWalk performs better than these approaches when they are given 90% of the data. The performance of SpectralClustering proves much more competitive, but DeepWalk still outperforms when labeled data is sparse on both Macro-F1 ($T_R$ ≤ 20%) and Micro-F1 ($T_R$ ≤ 60%).

This strong performance when only small fractions of the graph are labeled is a core strength of our approach. In the following experiments, we investigate the performance of our representations on even more sparsely labeled graphs.

> 在这个实验中，我们将 BlogCatlog network 上的训练比率($T_R$) 从 10% 提高到 90%。我们的结果显示在 表2 中。粗体数字代表每个列中的最佳表现。
>
> DeepWalk的表现始终优于 EdgeCluster、Modularity 和 wvRN。事实上，当只使用 20% 标记的节点进行训练时，DeepWalk比这些方法在获得 90% 的数据时表现更好。SpectralClustering 的表现被证明更具竞争力，但DeepWalk在标记数据在 Macro-F1 ($T_R \le 20\%$) 和 Micro-F1 ($T_R \le 60\%$) 稀疏的情况下仍然表现出色。
>
> 当只有一小部分顶点被标记时，这种优秀的表现是我们方法的核心优势。在接下来的实验中，我们将研究我们的表示在更稀疏标记的图上的性能。

#### 6.1.2 Flickr

In this experiment we vary the training ratio ($T_R$) on the Flickr network from 1% to 10%. This corresponds to having approximately 800 to 8,000 nodes labeled for classification in the entire network. Table 3 presents our results, which are consistent with the previous experiment. DeepWalk outperforms all baselines by at least 3% with respect to MicroF1. Additionally, its Micro-F1 performance when only 3% of the graph is labeled beats all other methods even when they have been given 10% of the data. In other words, DeepWalk can outperform the baselines with 60% less training data. It also performs quite well in Macro-F1, initially performing close to SpectralClustering, but distancing itself to a 1% improvement.

> 在本实验中，我们将Flickr网络上的训练比率 ($T_R$)从 1% 变到 10%。这对应于在整个网络中有大约 800 到 8000个节点被标记用于分类。Table3 给出了我们的结果，这与之前的实验是一致的。与 MicroF1 相比，DeepWalk 的表现至少比所有 baseline 高出 3%。此外，当只有 3% 的顶点被标记时，它的Micro-F1性能优于所有其他方法，即使它们已经被给予了 10% 的数据。换句话说，DeepWalk可以在训练数据减少 60% 的情况下超越 baseline。它在Macro-F1 中的表现也相当不错，最初的表现接近 SpectralCluging，但与 1% 的改进相去甚远。

#### 6.1.3 YouTube

The YouTube network is considerably larger than the previous ones we have experimented on, and its size prevents two of our baseline methods (SpectralClustering and Modularity) from running on it. It is much closer to a real world graph than those we have previously considered.

The results of varying the training ratio ($T_R$) from 1% to 10% are presented in Table 4. They show that DeepWalk significantly outperforms the scalable baseline for creating graph representations, EdgeCluster. When 1% of the labeled nodes are used for test, the Micro-F1 improves by 14%. The Macro-F1 shows a corresponding 10% increase. This lead narrows as the training data increases, but DeepWalk ends with a 3% lead in Micro-F1, and an impressive 5% improvement in Macro-F1.

This experiment showcases the performance benefits that can occur from using social representation learning for multilabel classification. DeepWalk, can scale to large graphs, and performs exceedingly well in such a sparsely labeled environment.

> YouTube网络比我们之前实验的 network 要大得多，而且它的大小使得我们的两个 baseline 方法(SpectralClusters 和 Modulality)无法在其上运行。它比我们之前考虑的更接近真实世界的图结构。
>
> 表4 列出了将训练比率（$T_R$）从 1% 更改为 10% 的结果。它们显示，DeepWalk 的性能显著优于创建图表示的可扩展 baseline EdgeCluster。当1% 的标记节点用于测试时，Micro-F1提高了 14%。Macro-F1 相应地提高了 10%。随着训练数据的增加，这一领先优势会缩小，但DeepWalk最终在Micro-F1中领先3%，在Macro-F1中提高了5%，令人印象深刻。
>
> 本实验展示了使用 social 表征学习进行多标签分类所能带来的表现效益。DeepWalk 可以扩展到大型图，并且在这样一个稀疏标记的环境中执行得非常好。

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Table2.png)

**Table 2: Multi-label classification results in BlogCatalog**

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Table3.png)

**Table 3: Multi-label classification results in Flickr**

![Table4](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Table4.png)

**Table 4: Multi-label classification results in YouTube**

## 6.2 Parameter Sensitivity

In order to evaluate how changes to the parameterization of DeepWalk effect its performance on classification tasks, we conducted experiments on two multi-label classifications tasks (Flickr, and BlogCatalog). In the interest of brevity, we have fixed the window size and the walk length to emphasize local structure ($w = 10, t = 40$). We then vary the number of latent dimensions ($d$), the number of walks started per vertex ($\gamma$), and the amount of training data available ($T_R$) to determine their impact on the network classification performance.

> 为了评估 DeepWalk 的参数化变化对其分类任务表现的影响，我们在两个多标签分类任务(Flickr和BlogCatalog)上进行了实验。为简洁起见，我们固定了窗口大小和步长，以强调局部结构($w=10，t=40$)。然后，我们改变潜在维数 ($d$)、每个顶点开始行走的次数($\gamma$)和可用的训练数据量($T_R$)来确定它们对 network 分类表现的影响。

#### 6.2.1 Effect of Dimensionality

Figure 5a shows the effects of increasing the number of latent dimensions available to our model.

Figures 5a1 and 5a3 examine the effects of varying the dimensionality and training ratio. The performance is quite consistent between both Flickr and BlogCatalog and show that the optimal dimensionality for a model is dependent on the number of training examples. (Note that 1% of Flickr has approximately as many labeled examples as 10% of BlogCatalog).

Figures 5a2 and 5a4 examine the effects of varying the dimensionality and number of walks per vertex. The relative performance between dimensions is relatively stable across different values of $\gamma$. These charts have two interesting observations. The first is that there is most of the benefit is accomplished by starting $\gamma = 30$ walks per node in both graphs. The second is that the relative difference between different values of $\gamma$ is quite consistent between the two graphs. Flickr has an order of magnitude more edges than BlogCatalog, and we find this behavior interesting.

These experiments show that our method can make useful models of various sizes. They also show that the performance of the model depends on the number of random walks it has seen, and the appropriate dimensionality of the model depends on the training examples available.

![Figure5](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Fig5.png)

**Figure 5: Parameter Sensitivity Study**

> 图5a显示了增加模型可用的潜在维度数量的效果。
>
> 图5a1和5a3查看了改变维度和训练比率的效果。Flickr 和 BlogCatalog 的表现相当一致，表明模型的最优维数取决于训练样本的数量。(请注意，1%的Flickr 标签数量相当于大约 10% 的 BlogCatalog 的标签数量)。
>
> 图5a2 和 5a4 查看改变维度和每个顶点的游走的次数的影响。在 $\gamma$ 的不同值上维度的表现相对稳定。这些图表有两个有趣的观察结果。第一，通过在两个图中的每个节点开始 $\gamma=30$ 次遍历 ，即可获得最大程度的收益。第二个是不同 $\gamma$ 的不同值之间的相对差异在两个图之间非常一致。Flickr的边比BlogCatalog多了一个数量级，我们发现这种表现很有趣。
>
> 这些实验表明，我们的方法可以生成有用的各种不同规模的模型。他们还表明，模型的性能取决于它所看到的随机游走的次数，而模型的合适维度取决于可用的训练样本。

#### 6.2.2 Effect of sampling frequency

Figure 5b shows the effects of increasing $\gamma$, the number of random walks that we start from each vertex.

The results are very consistent for different dimensions (Fig. 5b1, Fig. 5b3) and the amount of training data (Fig. 5b2, Fig. 5b4). Initially, increasing $\gamma$ has a big effect in the results, but this effect quickly slows ($\gamma > 10$). These results demonstrate that we are able to learn meaningful latent representations for vertices after only a small number of random walks.

> 图5b显示了增加$\Gamma$的效果，即我们从每个顶点开始的随机游走的次数。
>
> 对于不同的维度(图5b1，图5b3)和数据量(图5b2，图5b4)，结果非常一致。最初，增加 $\gamma$ 会对结果产生很大影响，但此效果会很快减慢($\gamma>10$)。这些结果表明，只需少量的随机游走，我们就能够学习到有意义的顶点潜在表示。

## 7. RELATED WORK

The main differences between our proposed method and previous work can be summarized as follows: 

1. We learn our latent social representations, instead of computing statistics related to centrality [13] or partitioning [41].
2. We do not attempt to extend the classification procedure itself (through collective inference [37] or graph kernels [21]).
3. We propose a scalable online method which uses only local information. Most methods require global information and are offline [17, 39–41].
4. We apply unsupervised representation learning to graphs. 

In this section we discuss related work in network classification and unsupervised feature learning.

> 我们提出的方法与以往工作的主要区别如下:
>
> 1. 我们学习 潜在的社会表征，而不是计算与中心性[13]或分割[41]相关的统计数据。
> 2. 我们不试图扩展分类过程本身(通过 collective inference [37]或  graph kernels [21])。
> 3. 我们提出了一种只使用局部信息的可扩展在线方法。大多数方法需要全局信息并且是离线的[17 ， 39-41]。
> 4. 我们将无监督表示学习应用于图。
>
> 在本节中，我们将讨论网络分类和无监督特征学习的相关工作。

## 7.1 Relational Learning

Relational classification (or collective classification) methods [15, 25, 32] use links between data items as part of the classification process. Exact inference in the collective classification problem is NP-hard, and solutions have focused on the use of approximate inference algorithm which may not be guaranteed to converge [37].

The most relevant relational classification algorithms to our work incorporate community information by learning clusters [33], by adding edges between nearby nodes [14], by using PageRank [24], or by extending relational classification to take additional features into account [43]. Our work takes a substantially different approach. Instead of a new approximation inference algorithm, we propose a procedure which learns representations of network structure which can then be used by existing inference procedure (including iterative ones).

A number of techniques for generating features from graphs have also been proposed [13, 17, 39–41]. In contrast to these methods, we frame the feature creation procedure as a representation learning problem.

Graph Kernels [42] have been proposed as a way to use relational data as part of the classification process, but are quite slow unless approximated [20]. Our approach is complementary; instead of encoding the structure as part of a kernel function, we learn a representation which allows them to be used directly as features for any classification method.

> 关系分类(或集合分类)方法[15,25,32]使用数据项之间的链接作为分类过程的一部分。在集合分类问题中，精确推理是 np-hard的，解决方法主要集中在使用不一定收敛[37]的近似推理算法。
>
> 我们工作中最相关的关系分类算法通过学习簇[33]、在邻近节点[14]之间添加边、使用 PageRank [24]或通过扩展关系分类以考虑额外特征[43]来整合 social 信息。我们的工作采用了完全不同的方法。我们提出了一种学习 network 结构表示的过程，而不是一种新的近似推理算法，它可以用于现有的推理过程(包括迭代的推理过程)。
>
> 许多从图中生成特征的技术也被提出[13,17,39-41]。与这些方法不同的是，我们将特征创建过程定义为一个表示学习问题。
>
> Graph Kernels[42]已经被提出作为一种使用关系数据作为分类过程的一部分的方法，但是它非常慢，除非近似[20]。我们的方法是互补的;我们不是将结构编码为核函数的一部分，而是学习一种表示，它允许它们直接作为任何分类方法的特征使用。

## 7.2 Unsupervised Feature Learning

Distributed representations have been proposed to model structural relationship between concepts [18]. These representations are trained by the back-propagation and gradient descent. Computational costs and numerical instability led to these techniques to be abandoned for almost a decade. Recently, distributed computing allowed for larger models to be trained [4], and the growth of data for unsupervised learning algorithms to emerge [10]. Distributed representations usually are trained through neural networks, these networks have made advancements in diverse fields such as computer vision [22], speech recognition [8], and natural language processing [1, 7].

> 已经提出了分布式表示来对概念之间的结构关系进行建模[18]。这些表示通过反向传播和梯度下降来训练。计算成本和数值不稳定导致这些技术被放弃了近十年。最近，分布式计算允许训练更大的模型[4]，并且出现了无监督学习算法的数据增长[10]。分布式表示通常通过神经网络进行训练，这些网络在不同的领域取得了进展，如计算机视觉[22]、语音识别[8]和自然语言处理[1,7]。

## 8. CONCLUSIONS

We propose DeepWalk, a novel approach for learning latent social representations of vertices. Using local information from truncated random walks as input, our method learns a representation which encodes structural regularities. Experiments on a variety of different graphs illustrate the effectiveness of our approach on challenging multi-label classification tasks.

As an online algorithm, DeepWalk is also scalable. Our results show that we can create meaningful representations for graphs which are too large for standard spectral methods. On such large graphs, our method significantly outperforms other methods designed to operate for sparsity. We also show that our approach is parallelizable, allowing workers to update different parts of the model concurrently.

In addition to being effective and scalable, our approach is also an appealing generalization of language modeling. This connection is mutually beneficial. Advances in language modeling may continue to generate improved latent representations for networks. In our view, language modeling is actually sampling from an unobservable language graph. We believe that insights obtained from modeling observable graphs may in turn yield improvements to modeling unobservable ones.

Our future work in the area will focus on investigating this duality further, using our results to improve language modeling, and strengthening the theoretical justifications of the method.

> 我们提出了 DeepWalk，一种学习顶点潜在社会表征的新方法。利用截断的随机游动的局部信息作为输入，我们的方法学习一个编码了结构规律的表示。在各种不同图上的实验证明了我们的方法在具有挑战性的多标签分类任务中的有效性。
>
> 作为一种在线算法，DeepWalk也是可扩展的。我们的结果表明，对于标准 的 spectral method来说太大的图，我们可以创建有意义的表示。在这样大的图上，我们的方法明显优于其他为稀疏性而设计的方法。我们还展示了我们的方法是可并行的，允许 worker 并发地更新模型的不同部分。
>
> 除了有效性和可扩展性之外，我们的方法也是一种的语言建模推广方法。这种联系是互惠互利的。语言建模的进步可能会继续为网络产生改进的潜在表示。在我们角度来看，语言建模实际上是从不可观察的语言图中采样的。我们相信，从对可观察图建模中获得的简介可能会反过来改进对不可观察图的建模。 
>
> 我们未来在该领域的工作将集中在进一步研究这种二重性，利用我们的结果来改进语言建模，并加强该方法的理论依据。

## 9. REFERENCES

[1] R. Al-Rfou, B. Perozzi, and S. Skiena. Polyglot: Distributed word representations for multilingual nlp. In Proceedings of the Seventeenth Conference on Computational Natural Language Learning, pages 183–192, Sofia, Bulgaria, August 2013. ACL. 

[2] R. Andersen, F. Chung, and K. Lang. Local graph partitioning using pagerank vectors. In Foundations of Computer Science, 2006. FOCS’06. 47th Annual IEEE Symposium on, pages 475–486. IEEE, 2006.

[3] Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review and new perspectives. 2013. 

[4] Y. Bengio, R. Ducharme, and P. Vincent. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155, 2003. 

[5] L. Bottou. Stochastic gradient learning in neural networks. In Proceedings of Neuro-Nˆımes 91, Nimes, France, 1991. EC2. 

[6] V. Chandola, A. Banerjee, and V. Kumar. Anomaly detection: A survey. ACM Computing Surveys (CSUR), 41(3):15, 2009. 

[7] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th ICML, ICML ’08, pages 160–167. ACM, 2008. 

[8] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1):30–42, 2012. 

[9] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. Le, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Ng. Large scale distributed deep networks. In P. Bartlett, F. Pereira, C. Burges, L. Bottou, and K. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1232–1240. 2012. 

[10] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. Why does unsupervised pre-training help deep learning? The Journal of Machine Learning Research, 11:625–660, 2010. 

[11] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A library for large linear classification. Journal of Machine Learning Research, 9:1871–1874, 2008. 

[12] F. Fouss, A. Pirotte, J.-M. Renders, and M. Saerens. Random-walk computation of similarities between nodes of a graph with application to collaborative recommendation. Knowledge and Data Engineering, IEEE Transactions on, 19(3):355–369, 2007. 

[13] B. Gallagher and T. Eliassi-Rad. Leveraging label-independent features for classification in sparselylabeled networks: An empirical study. In Advances in Social Network Mining and Analysis, pages 1–19. Springer, 2010. 

[14] B. Gallagher, H. Tong, T. Eliassi-Rad, and C. Faloutsos. Using ghost edges for classification in sparsely labeled networks. In Proceedings of the 14th ACM SIGKDD, KDD ’08, pages 256–264, New York, NY, USA, 2008. ACM. 

[15] S. Geman and D. Geman. Stochastic relaxation, gibbs distributions, and the bayesian restoration of images. Pattern Analysis and Machine Intelligence, IEEE Transactions on, (6):721–741, 1984. 

[16] L. Getoor and B. Taskar. Introduction to statistical relational learning. MIT press, 2007. 

[17] K. Henderson, B. Gallagher, L. Li, L. Akoglu, T. Eliassi-Rad, H. Tong, and C. Faloutsos. It’s who you know: Graph mining using recursive structural features. In Proceedings of the 17th ACM SIGKDD, KDD ’11, pages 663–671, New York, NY, USA, 2011. ACM. 

[18] G. E. Hinton. Learning distributed representations of concepts. In Proceedings of the eighth annual conference of the cognitive science society, pages 1–12. Amherst, MA, 1986. 

[19] R. A. Hummel and S. W. Zucker. On the foundations of relaxation labeling processes. Pattern Analysis and Machine Intelligence, IEEE Transactions on, (3):267–287, 1983. 

[20] U. Kang, H. Tong, and J. Sun. Fast random walk graph kernel. In SDM, pages 828–838, 2012.

[21] R. I. Kondor and J. Lafferty. Diffusion kernels on graphs and other discrete input spaces. In ICML, volume 2, pages 315–322, 2002. 

[22] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, volume 1, page 4, 2012. 

[23] D. Liben-Nowell and J. Kleinberg. The link-prediction problem for social networks. Journal of the American society for information science and technology, 58(7):1019–1031, 2007. 

[24] F. Lin and W. Cohen. Semi-supervised classification of network data using very few labels. In Advances in Social Networks Analysis and Mining (ASONAM), 2010 International Conference on, pages 192–199, Aug 2010. 

[25] S. A. Macskassy and F. Provost. A simple relational classifier. In Proceedings of the Second Workshop on Multi-Relational Data Mining (MRDM-2003) at KDD-2003, pages 64–76, 2003. 

[26] S. A. Macskassy and F. Provost. Classification in networked data: A toolkit and a univariate case study. The Journal of Machine Learning Research, 8:935–983, 2007. 

[27] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013. 

[28] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26, pages 3111–3119. 2013.

[29] T. Mikolov, W.-t. Yih, and G. Zweig. Linguistic regularities in continuous space word representations. In Proceedings of NAACL-HLT, pages 746–751, 2013. 

[30] A. Mnih and G. E. Hinton. A scalable hierarchical distributed language model. Advances in neural information processing systems, 21:1081–1088, 2009. 

[31] F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model. In Proceedings of the international workshop on artificial intelligence and statistics, pages 246–252, 2005. 

[32] J. Neville and D. Jensen. Iterative classification in relational data. In Proc. AAAI-2000 Workshop on Learning Statistical Models from Relational Data, pages 13–20, 2000. 

[33] J. Neville and D. Jensen. Leveraging relational autocorrelation with latent group models. In Proceedings of the 4th International Workshop on Multi-relational Mining, MRDM ’05, pages 49–55, New York, NY, USA, 2005. ACM. 

[34] J. Neville and D. Jensen. A bias/variance decomposition for models using collective inference. Machine Learning, 73(1):87–106, 2008. 

[35] M. E. Newman. Modularity and community structure in networks. Proceedings of the National Academy of Sciences, 103(23):8577–8582, 2006. 

[36] B. Recht, C. Re, S. Wright, and F. Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In Advances in Neural Information Processing Systems 24, pages 693–701. 2011. 

[37] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93, 2008. 

[38] D. A. Spielman and S.-H. Teng. Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear systems. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing, pages 81–90. ACM, 2004. 

[39] L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM. 

[40] L. Tang and H. Liu. Scalable learning of collective behavior based on sparse social dimensions. In Proceedings of the 18th ACM conference on Information and knowledge management, pages 1107–1116. ACM, 2009. 

[41] L. Tang and H. Liu. Leveraging social media networks for classification. Data Mining and Knowledge Discovery, 23(3):447–478, 2011. 

[42] S. Vishwanathan, N. N. Schraudolph, R. Kondor, and K. M. Borgwardt. Graph kernels. The Journal of Machine Learning Research, 99:1201–1242, 2010. 

[43] X. Wang and G. Sukthankar. Multi-label relational neighbor classification using social context features. In Proceedings of the 19th ACM SIGKDD, pages 464–472. ACM, 2013. 

[44] W. Zachary. An information flow model for conflict and fission in small groups1. Journal of anthropological research, 33(4):452–473, 1977.