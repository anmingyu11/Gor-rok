## 5.Experiments

In this section, we experimentally verify our analysis and evaluate the performance of different tree models on both synthetic and real data. Throughout experiments, we use OTM to denote the tree model trained according to Algorithm 1 since its goal is to learn optimal tree models under beam search. To perform an ablation study, we consider two variants of OTM: OTM (-BS) differs from OTM by replacing $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ with $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \cup \mathcal{S}_h^{-}(\mathbf{y})$ , and OTM (-OptEst) differs from OTM by replacing $\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ in Eq. (13) with $z_n$ in Eq. (1). More details of experiments can be found in the supplementary materials.

> 在本节中，我们通过实验证明了我们的分析，并评估了不同树模型在合成数据和真实数据上的性能。在整个实验过程中，我们使用OTM来表示根据算法1训练的树模型，因为它的目标是在 beam search 下学习最优的树模型。为了进行消融研究，我们考虑了OTM的两个变体：OTM (-BS)与OTM不同之处在于将 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 替换为 $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \cup \mathcal{S}_h^{-}(\mathbf{y})$；OTM (-OptEst)与OTM不同之处在于将公式（13）中的 $\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 替换为公式（1）中的 $z_n$。更多实验的详细信息可以在附录中找到。

### 5.1. Synthetic Data

**Datasets**: For each instance $(\mathbf{x}, \mathbf{y})$, $\mathbf{x} \in \mathbb{R}^d$ is sampled from a d-dimensional isotropic Gaussian distribution $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$ with zero mean and identity covariance matrix, and $\mathbf{y} \in \{0,1\}^M$ is sampled from $p(\mathbf{y} \mid \mathbf{x})=\prod_{j=1}^M p\left(y_j \mid \mathbf{x}\right)=\prod_{j=1}^M 1 /\left(1+\exp \left(-\left(2 y_j-1\right) \mathbf{w}_j^{\top} \mathbf{x}-b\right)\right)$ where the weight vector $\mathbf{w}_j \in \mathbb{R}^d$ is also sampled from $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$. The bias $b$ is a predefined constant9 to control the number of non-zero entries in $\mathbf{y}$. Corresponding training and testing datasets are denoted as $D_{tr}$ and $D_{te}$, respectively.

**Compared Models and Metric**: We compare OTM with PLT and TDM. All the tree models $\mathcal{M}(\mathcal{T}, g)$ share the same tree structure $\mathcal{T}$ and the same parameterization of the node wise scorer $g$. More specifically, $\mathcal{T}$ is set to be a random binary tree over $\mathcal{I}$ and $g(\mathbf{x}, n)=\boldsymbol{\theta}_n^{\top} \mathbf{x}+b_n$ is parameterized as a linear scorer, where $\boldsymbol{\theta}_n \in \mathbb{R}^d$ and $b_n \in \mathbb{R}$ are trainable parameters. All models are trained on $D_{tr}$ and their perfomance is measured by $\widehat{\operatorname{reg}}_{p @ m}$, which is an estimation of $\operatorname{reg}_{p @ m}(\mathcal{M})$ defined in Eq. (10) by replacing the expectation over $p(\mathbf{x})$ with the summation over $(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{t e}$.

**Results**: Table 2 shows that OTM performs the best compared to other models, which indicates that eliminating the training-testing discrepancy can improve retrieval performance of tree models. Both OTM (-BS) and OTM (-OptEst) have smaller regret than PLT and TDM, which means that using beam search aware subsampling (i.e., $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$) or estimated optimal pseudo targets (i.e., $\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$) alone contributes to better performance. Besides, OTM (-OptEst) has smaller regret than OTM (-BS), which reveals that beam search aware subsampling contributes more than estimated optimal pseudo targets to the performance of OTM.

> **数据集**：对于每个实例 $(\mathbf{x}, \mathbf{y})$，$\mathbf{x} \in \mathbb{R}^d$ 是从零均值和单位协方差矩阵的 $d$ 维各向同性高斯分布 $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$ 中采样得到的，而 $\mathbf{y} \in \{0,1\}^M$ 是从 $p(\mathbf{y} \mid \mathbf{x})=\prod_{j=1}^M p\left(y_j \mid \mathbf{x}\right)=\prod_{j=1}^M 1 /\left(1+\exp \left(-\left(2 y_j-1\right) \mathbf{w}_j^{\top} \mathbf{x}-b\right)\right)$ 中采样得到的，其中权重向量 $\mathbf{w}_j \in \mathbb{R}^d$ 也是从 $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$ 中采样得到的。偏置 $b$ 是一个预定义的常数，用于控制 $\mathbf{y}$ 中非零条目的数量。相应的训练和测试数据集分别表示为 $D_{tr}$ 和 $D_{te}$。
>
> **比较的模型和指标**：我们将OTM与PLT和TDM进行比较。所有的树模型 $\mathcal{M}(\mathcal{T}, g)$ 共享相同的树结构 $\mathcal{T}$ 和节点评分器 $g$ 的参数化。具体而言，$\mathcal{T}$ 被设置为在 $\mathcal{I}$ 上的随机二叉树，$g(\mathbf{x}, n)=\boldsymbol{\theta}_n^{\top} \mathbf{x}+b_n$ 被参数化为线性评分器，其中 $\boldsymbol{\theta}_n \in \mathbb{R}^d$ 和 $b_n \in \mathbb{R}$ 是可训练的参数。所有模型都在 $D_{tr}$ 上进行训练，并通过 $\widehat{\operatorname{reg}}_{p @ m}$ 来衡量它们的性能，该指标是将公式（10）中对 $p(\mathbf{x})$ 的期望替换为对 $\mathcal{D}_{t e}$ 中 $(\mathbf{x}, \mathbf{y})$ 的求和得到的。
>
> **结果**：表2显示OTM相比于其他模型表现最好，这表明消除训练-测试差异可以提高树模型的检索性能。OTM (-BS)和OTM (-OptEst)的遗憾值比PLT和TDM小，这意味着仅使用 beam search 感知子采样（即$\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$）或估计的最优伪目标（即$\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$）单独对性能有所贡献。此外，OTM (-OptEst)的遗憾值比OTM (-BS)小，这表明 beam search 感知子采样对OTM的性能贡献大于估计的最优伪目标。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table2.png)

**Table 2**. A comparison of $\widehat{\operatorname{reg}}_{p @ m}(\mathcal{M})$ averaged by 5 runs with random initialization with hyperparameter settings $M = 1000$, $d = 10$, $b = −5$, $|D_{tr}| = 10000$, $|D_{te}| = 1000$ and $k = 50$.

### 5.2. Real Data

**Datasets**: Our experiment are conducted on two large-scale real datasets for recommendation tasks: Amazon Books (McAuley et al., 2015; He & McAuley, 2016) and UserBehavior (Zhu et al., 2018). Each record of both datasets is organized in the format of user-item interaction, which contains user ID, item ID and timestamp. The original interaction records are formulated as a set of user-based data. Each user-based data is denoted as a list of items sorted by the timestep that the user-item interaction occurs. We discard the user based data which has less than 10 items and split the rest into training set $D_{tr}$, validation set $D_{val}$ and testing set $D_{te}$ in the same way as Zhu et al. (2018; 2019). For the validation and testing set, we take the first half of each user-based data according to ascending order along timestamp as the feature $\mathbf{x}$, and the latter half as the relevant targets $y$. While training instances are generated from the raw user-based data considering the characteristics of different approaches on the training set. If the approach restricts $\left|\mathcal{I}_{\mathbf{x}}\right|=1$, we use a sliding window to produce several instances for each user based data, while one instanceis obtained for methods without restriction on $|\mathcal{I}_{\mathbf{x}}|$. 

**Compared Models and Metric**: We compare OTM with two series of methods: (1) widely used methods in recommendation tasks, such as Item-CF (Sarwar et al., 2001), the basic collaborative filtering method, and YouTube productDNN (Covington et al., 2016), the representative work of vector kNN based methods; (2) tree models like HSM (Morin & Bengio, 2005), PLT and JTM (Zhu et al., 2019). HSM is a hierarchical softmax model which can be regarded as PLT with the $\left|\mathcal{I}_{\mathbf{x}}\right|=1$ restriction. JTM is a variant of TDM which trains tree structure and node-wise scorers jointly and achieves state-of-the-art performance on these two datasets. All the tree models share the same binary tree structure and adopt the same neural network model for node-wise scorers. The neural network consists of three fully connected layers with hidden size 128, 64 and 24 and parametric ReLU is used as the activation function. The performance of different models is measured by Precision@m (Eq. (5)), Recall@m (Eq. (6)) and F-Measure@m (Eq. (7)) averaged over the testing set $D_{te}$.

**Results**: Table 3 and Table 4 show results of Amazon Books and UserBehavior, respectively10. Our model performs the best among all methods: Compared to the previous state-ofthe-art JTM, OTM achieves 29.8% and 6.3% relative recall lift (m = 200) on Amazon Books and UserBehavior separately. Results of OTM and its two variants are consistent with that on synthetic data: Both beam search aware subsampling and estimated optimal pseudo targets contribute to better performance, while the former contributes more and the performance of OTM mainly depends on the former. Besides, the comparison between HSM and PLT also demonstrates that removing the restriction of $\left|\mathcal{I}_{\mathbf{x}}\right|=1$ in tree models contributes to performance improvement.

>**数据集**：我们的实验是在两个大规模实际推荐任务数据集上进行的：Amazon图书数据集（McAuley等，2015；He＆McAuley，2016）和UserBehavior数据集（Zhu等，2018）。这两个数据集中的每条记录都以user-item 互动的方式组织，包含item ID、item ID和时间戳。原始的互动记录被表示为一组基于用户的数据。每个基于用户的数据被表示为按照 user-item 互动发生的时间步排序的 item 列表。我们丢弃了少于10个item 的基于用户的数据，并将其余数据集按照相同的方式分割为训练集$D_{tr}$、验证集$D_{val}$和测试集$D_{te}$。对于验证集和测试集，我们按照时间戳的升序顺序将每个基于用户的数据的前一半作为特征 $\mathbf{x}$，后一半作为相关目标 $y$。而训练实例是根据原始的基于用户的数据生成的，考虑到不同方法在训练集上的特点。如果方法限制$\left|\mathcal{I}_{\mathbf{x}}\right|=1$，我们使用一个滑动窗口为每个基于用户的数据生成多个实例，而对于没有$|\mathcal{I}_{\mathbf{x}}|$限制的方法，获得一个实例。
>
>**比较的模型和指标**：我们将OTM与两类方法进行比较：（1）推荐任务中广泛使用的方法，例如Item-CF（Sarwar等，2001），基本的协同过滤方法，以及基于向量kNN的代表性作品YouTube productDNN（Covington等，2016）；（2）树模型，如HSM（Morin＆Bengio，2005），PLT和JTM（Zhu等，2019）。HSM是一种分层softmax模型，可以看作是具有$\left|\mathcal{I}_{\mathbf{x}}\right|=1$限制的PLT。JTM是TDM的一种变体，它同时训练树结构和节点评分器，并在这两个数据集上取得了最先进的性能。所有的树模型共享相同的二叉树结构，并采用相同的神经网络模型作为节点评分器。神经网络包含三个全连接层，隐藏大小分别为128、64和24，并使用参数化ReLU作为激活函数。不同模型的性能通过在测试集$D_{te}$上对Precision@m（公式（5））、Recall@m（公式（6））和F-Measure@m（公式（7））进行平均来衡量。 
>
>**结果**：表3和表4分别显示了 Amazon图书 数据集和 UserBehavior数据集 的结果。我们的模型在所有方法中表现最好：与之前最先进的JTM相比，OTM在Amazon图书数据集和UserBehavior数据集上分别实现了29.8%和6.3%的相对召回率提升（m = 200）。OTM及其两个变体的结果与合成数据上的结果一致：beam search 感知子采样和估计的最优伪目标都对性能有所贡献，其中前者的贡献更大，OTM的性能主要依赖于前者。此外，HSM和PLT之间的比较也证明了在树模型中消除 $\left|\mathcal{I}_{\mathbf{x}}\right|=1$ 限制有助于提高性能。

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table3.png)

**Table 3**. Precision@m, Recall@m and F-Measure@m comparison on Amazon Books with beam size k = 400 and various m (%).

> **表3**。在Amazon图书数据集上，使用 beam search 大小 k = 400 和不同 m（%）的情况下，对Precision@m、Recall@m和F-Measure@m进行比较。

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table4.png)

**Table 4**. Precision@m, Recall@m and F-Measure@m comparison on UserBehavior with beam size k = 400 and various m (%).

> **表格4**。在UserBehavior数据集上，使用 beam search 大小 k = 400 和不同 m（%）的情况下，对Precision@m、Recall@m和F-Measure@m进行比较。

To understand why OTM achieves more significant improvement (29.8% versus 6.3%) on Amazon Books than UserBehavior, we analyze the statistics of these datasets and their corresponding tree structure. For each $n ∈ \mathcal{N}$ , we define $S_n=\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{t r}} z_n$ to count the number of training instances which are relevant to $n$ (i.e., $z_n=1$ ). For each level $1 ≤ h ≤ H$, we sort $\left\{S_n: n \in \mathcal{N}_h\right\}$ in a descending order and normalize them as $S_n / \sum_{n^{\prime} \in \mathcal{N}_h} S_{n^{\prime}}$ . This produces a level-wise distribution, which reflects the data imbalance on relevant nodes resulted from the intrinsic property of both the datasets and the tree structure. As is shown in Figure 2, the level-wise distribution of UserBehavior has a heavier tail than that of Amazon Books at the same level. This implies the latter has a higher proportion of instances concentrated on only parts of nodes, which makes it easier for beam search to retrieve relevant nodes for training and thus leads to more significant improvement.

To verify our analysis on the time complexity of tree models, we compare their empirical training time, since they share the same beam search process in testing. More specifically, we compute the wall-clock time per batch for training PLT, TDM and PLT with batch size 100 on the UserBehavior dataset. This number is averaged over 5000 training iterations on a single Tesla P100-PCIE-16GB GPU. The results are 0.184s for PLT, 0.332s for TDM and 0.671s for OTM, respectively. Though OTM costs longer time than PLT and JTM, they have the same order of magnitude. This is not weird, since the step 4 and 5 in Algorithm 1 only increases the constant factor of complexity. Besides, this is a reasonable trade-off for better performance and distributed training can alleviate this in practical applications.

> 为了理解为什么OTM在Amazon图书数据集上实现了比UserBehavior更显著的提升（29.8%对比6.3%），我们分析了这些数据集及其对应的树结构的统计信息。对于每个$n \in \mathcal{N}$，我们定义 $S_n=\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{tr}} z_n$ 来计算与 $n$ 相关的训练实例的数量（即$z_n=1$）。对于每个层级 $1 \leq h \leq H$，我们将 $\left\{S_n: n \in \mathcal{N}_h\right\}$ 按降序排序，并将它们归一化为 $S_n / \sum_{n^{\prime} \in \mathcal{N}_h} S_{n^{\prime}}$。这产生了一个按层级划分的分布，反映了由于数据集和树结构的固有属性而导致的相关节点上的数据不平衡情况。如图2所示，UserBehavior的按层级分布比Amazon图书更加尾重。这意味着后者有更高比例的实例集中在部分节点上，这使得 beam search 更容易检索到用于训练的相关节点，从而导致更显著的改进。 
>
> 为了验证我们关于树模型时间复杂度的分析，我们比较了它们的实证训练时间，因为它们在测试中共享相同的 beam search 过程。具体来说，我们计算了在UserBehavior数据集上使用批量大小为100进行训练的PLT、TDM和PLT的每个批次的墙钟时间。这个数字是在一块Tesla P100-PCIE-16GB GPU上的5000个训练迭代中平均得出的。结果分别为0.184秒的PLT，0.332秒的TDM和0.671秒的OTM。尽管OTM的时间比PLT和JTM更长，但它们具有相同数量级的时间复杂度。这并不奇怪，因为算法1中的第4步和第5步只增加了复杂性的常数因子。此外，这是为了获得更好的性能的合理权衡，在实际应用中可以通过分布式训练来减轻这个问题。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Figure2.png)

**Figure 2**. Results of the level-wise distribution versus node index on Amazon Books and UserBehavior with $h = 8 (|\mathcal{N}_h| = 256)$.

> **图2**。Amazon图书和UserBehavior数据集上按层级分布与节点索引的结果，其中$h = 8$（$|\mathcal{N}_h| = 256$）。

## 6. Conclusions and Future Work

Tree models have been widely adopted in large-scale information retrieval and recommendation tasks due to their logarithmic computational complexity. However, little attention has been paid to the training-testing discrepancy where the retrieval performance deterioration caused by beam search in testing is ignored in training. To the best of our knowledge, we are the first to study this problem on tree models theoretically. We also propose a novel training algorithm for learning optimal tree models under beam search which achieves improved experiment results compared to the state-of-the-arts on both synthetic and real data.

For future work, we’d like to explore other techniques for training g(x, n) according to Eq. (14), e.g., the REINFORCE algorithm (Williams, 1992; Ranzato et al., 2016) and the actor-critic algorithm (Sutton et al., 2000; Bahdanau et al., 2017). We also want to extend our algorithm for learning tree structure and node-wise scorers jointly. Besides, applying our algorithm to applications like extreme multilabel text classification is also an interesting direction.

> 树模型由于其对数级的计算复杂度，在大规模信息检索和推荐任务中得到了广泛应用。然而，对于训练-测试不一致性的问题，人们对训练中忽略了由于测试中的 beam search 导致的检索性能下降并没有给予足够重视。据我们所知，我们是第一个从理论上研究树模型中这个问题的人。我们还提出了一种新颖的训练算法，用于在 beam search 下学习最优的树模型，并在合成数据和真实数据上实现了比现有方法更好的实验结果。
>
> 对于未来的工作，我们希望探索其他根据方程（14）来训练$g(\mathbf{x}, n)$ 的技术，例如REINFORCE算法（Williams, 1992; Ranzato et al., 2016）和actor-critic算法（Sutton et al., 2000; Bahdanau et al., 2017）。我们还希望扩展我们的算法，同时学习树结构和节点评分器。此外，将我们的算法应用于极端多标签文本分类等应用也是一个有趣的方向。

## Acknowledgements

We deeply appreciate Xiang Li, Rihan Chen, Daqing Chang, Pengye Zhang, Jie He and Xiaoqiang Zhu for their insightful suggestions and discussions. We thank Huimin Yi, Yang Zheng, Siran Yang, Guowang Zhang, Shuai Li, Yue Song and Di Zhang for implementing the key components of the training platform. We thank Linhao Wang, Yin Yang, Liming Duan and Guan Wang for necessary supports about online serving. We thank anonymous reviewers for their constructive feedback and helpful comments.



## References

Bahdanau, D., Brakel, P., Xu, K., Goyal, A., Lowe, R., Pineau, J., Courville, A., and Bengio, Y. An actor-critic algorithm for sequence prediction. In International Conference on Learning Representations, 2017. 

Cohen, E. and Beck, C. Empirical analysis of beam search performance degradation in neural sequence models. In International Conference on Machine Learning, pp. 1290– 1299, 2019. 

Covington, P., Adams, J., and Sargin, E. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pp. 191–198, 2016. 

Daume III, H. and Marcu, D. Learning as search opti- ´ mization: Approximate large margin methods for structured prediction. In International Conference on Machine learning, pp. 169–176. ACM, 2005. 

Goyal, K., Neubig, G., Dyer, C., and Berg-Kirkpatrick, T. A continuous relaxation of beam search for end-to-end training of neural sequence models. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018. 

He, R. and McAuley, J. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pp. 507–517, 2016. 

Jain, H., Prabhu, Y., and Varma, M. Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 935–944, 2016.

Jasinska, K., Dembczynski, K., Busa-Fekete, R., Pfannschmidt, K., Klerx, T., and Hullermeier, E. Extreme f-measure maximization using sparse probability estimates. In International Conference on Machine Learning, pp. 1435–1444, 2016.

Khandagale, S., Xiao, H., and Babbar, R. Bonsai-diverse and shallow trees for extreme multi-label classification. arXiv preprint arXiv:1904.08249, 2019.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Kool, W., Van Hoof, H., and Welling, M. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In International Conference on Machine Learning, pp. 3499–3508, 2019.

Lapin, M., Hein, M., and Schiele, B. Analysis and optimization of loss functions for multiclass, top-k, and multilabel classification. IEEE transactions on pattern analysis and machine intelligence, 40(7):1533–1554, 2017.

McAuley, J., Targett, C., Shi, Q., and Van Den Hengel, A. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 43–52, 2015.

Menon, A. K., Rawat, A. S., Reddi, S., and Kumar, S. Multilabel reductions: what is my loss optimising? In Advances in Neural Information Processing Systems, pp. 10599–10610, 2019.

Morin, F. and Bengio, Y. Hierarchical probabilistic neural network language model. In Proceedings of the eighth international conference on artificial intelligence and statistics, volume 5, pp. 246–252. Citeseer, 2005.

Negrinho, R., Gormley, M., and Gordon, G. J. Learning beam search policies via imitation learning. In Advances in Neural Information Processing Systems, pp. 10652– 10661, 2018.

Prabhu, Y. and Varma, M. Fastxml: A fast, accurate and stable tree-classifier for extreme multi-label learning. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 263–272, 2014.

Prabhu, Y., Kag, A., Harsola, S., Agrawal, R., and Varma, M. Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising. In Proceedings of the 2018 World Wide Web Conference, pp. 993–1002. International World Wide Web Conferences Steering Committee, 2018.

Ranzato, M., Chopra, S., Auli, M., and Zaremba, W. Sequence level training with recurrent neural networks. In International Conference on Learning Representations, 2016.

Ross, S., Gordon, G., and Bagnell, D. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635, 2011.

Sarwar, B., Karypis, G., Konstan, J., and Riedl, J. Itembased collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web, pp. 285–295, 2001.

Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems, pp. 1057–1063, 2000.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256, 1992.

Wiseman, S. and Rush, A. M. Sequence-to-sequence learning as beam-search optimization. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1296–1306, 2016.

Wu, X.-Z. and Zhou, Z.-H. A unified view of multi-label performance measures. In International Conference on Machine Learning, pp. 3780–3788. JMLR. org, 2017.

Wydmuch, M., Jasinska, K., Kuznetsov, M., Busa-Fekete, R., and Dembczynski, K. A no-regret generalization of hierarchical softmax to extreme multi-label classification. In Advances in Neural Information Processing Systems, pp. 6355–6366, 2018.

Xu, Y. and Fern, A. On learning linear ranking functions for beam search. In International Conference on Machine learning, pp. 1047–1054, 2007.

Yang, F. and Koyejo, S. On the consistency of top-k surrogate losses. arXiv preprint arXiv:1901.11141, 2019.

You, R., Zhang, Z., Wang, Z., Dai, S., Mamitsuka, H., and Zhu, S. Attentionxml: Label tree-based attention-aware deep model for high-performance extreme multi-label text classification. In Advances in Neural Information Processing Systems, pp. 5812–5822, 2019.

Zhu, H., Li, X., Zhang, P., Li, G., He, J., Li, H., and Gai, K. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1079–1088. ACM, 2018.

Zhu, H., Chang, D., Xu, Z., Zhang, P., Li, X., He, J., Li, H., Xu, J., and Gai, K. Joint optimization of tree-based index and deep model for recommender systems. In Advances in Neural Information Processing Systems, pp. 3973–3982, 2019.