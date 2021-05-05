# Wide & Deep Learning for Recommender Systems

## ABSTRACT

具有非线性特征变换的广义线性模型被广泛应用于具有稀疏输入的大规模回归和分类问题。通过广泛的 cross- product 特征变换来记忆特征交叉是有效和可解释的，而泛化则需要更多的特征工程工作。特征工程较少的情况下，通过学习稀疏特征的 low-dimensional dense embedding ，深度神经网络可以更好地泛化到看不见的特征组合。然而，当 user 与 item 的交互是稀疏且 high-rank 时，有着 embedding 的深度神经网络会过度泛化并且推荐不太相关的产品。

在本文中，我们提出了 Wide & Deep learning - 联合训练 wide 线性模型 和 deep 神经网络- 将记忆性和泛化性的优点结合起来并应用于推荐系统。我们在 Google Play 上对该系统进行了生产和评估，Google Play 是一家商业移动应用商店，拥有超过10亿活跃用户和100多万个应用。线上实验结果显示，Wide&Deep 与 Wide-Only 和 Deep-Only 两种模式相比，显著增加了APP的采购量。我们还开源了我们在 TensorFlow 中的实现。

## 1. INTRODUCTION

推荐系统可以被视为搜索排序系统，其中输入查询是一组 user 和 context 信息，而输出是排序后的 item 列表。在给定查询的情况下，推荐任务是在数据库中查找相关 item，然后根据特定目标(例如 clicks 或 purchases)对 item 进行排序。

类似于一般搜索排序问题，推荐系统中的一项挑战是同时实现记忆性和泛化性。

- 记忆性可以粗略地定义为学习 item 或特征的频繁共现，并利用历史数据中可用的相关性。
- 泛化性基于相关性的传递性，并探索过去从未或很少出现的新特征组合。

**基于记忆性的推荐通常更具针对性，并与用户已经有过行为的 item 相关。与记忆性相比，泛化性更倾向于提高推荐的 item 的多样性。**在本文中，我们关注的是 Google Play 商店的应用程序推荐问题，但该方法应该适用于通用推荐系统。

对于工业背景下的大规模在线推荐和排序系统，广义线性模型(如逻辑回归)被广泛使用，因为它们简单、可扩展和可解释。该模型通常采用 one-hot 编码的二值稀疏特征进行训练。例如，如果用户安装了 netflix ，二值特征 **"user_installed_app=netflix"** 的值为 $1$。通过稀疏特征的 cross-product 转换可以有效地实现记忆性，例如 **AND(user_installed_app=netflix，impression_app=pandora ")**，如果用户安装了 netflix，然后显示 pandora，其值为 $1$。这解释了特征对的共现是如何与目标标签相关联的。泛化可以通过使用粒度更小的特性来添加，例如 **AND(user_installed_category=video，impression_category=music)**，但通常需要手工特征工程。cross-product 转换的一上限是它们不能泛化到没有出现在训练数据中的 query-item 特征对。

Embdding-based 的模型，如因子分解机[5]或深度神经网络，可以通过学习每个 query-item 特征的 low-dimensional dense embedding 向量来推广到以前未曾见过的 query-item 特征对，而特征工程的负担更小。

然而，当 query-item 矩阵是稀疏且高阶的(例如用户具有特定的偏好或小众)时，很难学习 query-item 的有效低维表示。在这种情况下，大多数 query-item 对之间是没有交互的，但是 dense embedding 将使得所有 query-item 对的预测为非零，从而可能过度泛化并给出了不相关的 item。另一方面，使用 cross-product 特征变换的线性模型可以用更少的参数记住这些 “额外规则”。

在本文中，我们提出了 Wide & Deep 学习框架，通过联合训练线性模型组件和神经网络组件，在一个模型中实现记忆性和泛化性，如图1所示。

论文的主要贡献包括：

- 针对输入稀疏的通用推荐系统，提出了一种用于联合训练 embedding 前馈神经网络和特征变换线性模型的 Wide&Deep 学习框架。
- W&D 推荐系统的实践和评估是在 Google Play 的生产环境，Google Play是一家移动应用商店，拥有超过10亿活跃用户和超过100万个应用程序。
- 我们已经将我们的实现开源，其中包括 TensorFlow1 中的高级 API 。

虽然想法很简单，但我们表明，Wide&Deep 框架在满足训练和服务速度要求的同时，显著提高了移动应用商店上的应用获取率。

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig1.png)

**Figure 1: The spectrum of Wide & Deep models**

## 2. RECOMMENDER SYSTEM OVERVIEW

图2 显示了推荐系统的应用概况。当用户访问应用商店时，会生成一个query，该 query 可以包括各种用户和上下文特征。推荐系统返回用户可以在其上执行诸如点击或购买等特定动作的 app 列表(也称为impressions)。这些用户动作以及 queries 和 impressions 被记录在日志中，作为学习者的训练数据。

由于数据库中有 100多万个应用程序，因此很难在服务延迟要求(通常为$O(10)$毫秒)内对每个 query 的每个 app 进行详尽的评分。因此，收到查询后的第一步是检索。检索系统使用各种信号(通常是机器学习的模型和人定义的规则的组合)返回与 query 最匹配的 item 的列表。

减少候选库规模后，排序系统会根据所有 item 的分数对其进行排序。分数通常是 $P(y|\mathbf{x})$、给定特征 $\mathbf{x}$ 的用户 action 标签 $y$ 的概率，所述特征 $\mathbf{x}$ 包括用户特征 (例如，国家、语言、人口统计) 、上下文特征 (例如，设备、每天的小时、星期几) 和 impression 特征 (例如，应用的年龄、应用的历史统计)。在本文中，我们重点研究了基于 W&D 学习框架的排序模型。

## 3. WIDE & DEEP LEARNING

#### 3.1 The Wide Component

Wide 部分是形式为 $y=\mathbf{w}^T\mathbf{x}+b$ 的广义线性模型，如图1(左)所示。$y$ 是预测值，$\mathbf{x}=[x_1，x_2，\cdots，x_d]$ 是 $d$ 特征的向量，$\mathbf{w}=[w_1，w_2，\cdots，w_d]$ 是模型参数，$b$ 是偏差。特征集合包括原始输入特征和变换后的特征。最重要的变换之一是 cross-product 变换，其定义为：
$$
\phi_{k}(\mathbf{x})=\prod_{i=1}^{d} x_{i}^{c_{k i}} 
\quad c_{k i} \in\{0,1\}
\qquad(1)
$$
其中，$c_{ki}$ 是布尔变量，如果第 $i$ 个特征是第 $k$ 个转换 $\phi_k$ 的一部分，则为 $1$ ，否则为 $0$ 。对于二元特征，当且仅当构成特征 (**“AND(gender=female, language=en)”**) 都为 $1$ 时，cross-product 变换为 $1$，否则为 $0$ (例如，**“AND(gender=female，language=en)”**)。这捕获了二值特征之间的交叉，并将非线性添加到广义线性模型中。

#### 3.2 The Deep Component

deep 部分是前馈神经网络，如图1(右)所示。对于类别特征，原始输入是特征字符串(例如，**“language=en”**)。这些稀疏的、high-dimensional categorical features 中的每一个首先被转换成 low-dimensional and dense 实值向量，通常被称为 embedding 向量。embedding 的维度通常在 $O(10)$ 到 $O(100)$ 的数量级上。在模型训练过程中，随机初始化 embedding 向量，然后训练这些值以最小化最终的损失函数。然后，这些低维的密集 embedding 向量在前向传播中被送到神经网络的隐藏层。具体地说，每个隐藏层执行以下计算：
$$
a^{(l+1)}=f\left(W^{(l)} a^{(l)}+b^{(l)}\right) \qquad(12)
$$
其中 $l$ 是层号，$f$ 是激活函数，通常是整流线性单元(RELU)。$a^{(L)}$、$b^{(L)}$ 和 $W^{(L)}$ 是第 $l$ 层的激活函数、偏置单元和模型权重。

#### 3.3 Joint Training of Wide & Deep Model

将 wide component 和 deep component 以其输出对数概率的加权和作为预测值进行组合，然后将其输入一个通用的 logistic 损失函数进行联合训练。注意联合训练和 ensemble 训练是有区别的。在 ensemble中，单个模型在互相独立的情况下独立训练，并且它们的预测只在推理时组合，而不是在训练时。而联合训练则通过在训练时同时考虑 Deep 和 Wide 部分以及它们之和的权重来同时优化所有参数。

对模型大小也有影响：对于一个 ensemble 模型，由于训练是不连贯的，每个独立的模型大小通常需要更大(例如，有更多的特征和特征变换)，以达到一个 ensemble 工作的合理精度。相比之下，对于联合训练，Wide 部分只需要通过少量的 cross-product 特征变换来弥补 Deep 部分的不足，而不需要一个全尺寸的 Wide 模型。

W&D 模型的联合训练是通过使用小批量随机优化将梯度从输出同时反向传播到模型的 Wide 和 Deep 两部分来完成的。

在实验中，我们使用带 L1 正则化的 Followthe-regular-leader(FTRL)算法[3]作为模型的 wide 部分的优化器，而 AdaGrad[1] 作为模型的 deep 部分的优化器。

组合模型如图1(中间)所示。对于逻辑回归问题，模型的预测为：
$$
P(Y=1 \mid \mathbf{x})=\sigma\left(\mathbf{w}_{w i d e}^{T}[\mathbf{x}, \phi(\mathbf{x})]+\mathbf{w}_{d e e p}^{T} a^{\left(l_{f}\right)}+b\right)
\qquad (3)
$$
其中 $Y$ 是二分类标签，$\sigma(·)$ 是 sigmoid 函数，$\phi(X)$ 是原始特征 $\mathbf{x}$ 的 cross-product 变换，$b$ 是偏置项。$\mathbf{w}_{wide}$ 是所有 wide 模型权重的向量，$\mathbf{w}_{deep}$ 是应用于最终激活 $a^{(l_f)}$ 的权重。

## 4. SYSTEM IMPLEMENTATION

APPS 推荐流程的实现包括三个阶段：数据生成、模型训练和模型服务，如 图3 所示。

![Figure3](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig3.png)

**Figure 3: Apps recommendation pipeline overview.**

#### 4.1 Data Generation

在这个阶段，我们使用一段时间内的 user 和 app impression 数据来生成训练数据。每个例子对应一个 impression。标签是应用是否被获取：如果应用被安装，则为 $1$，否则为 $0$。

词汇表是将类别特征映射到 id 的表，也在这个阶段生成。系统为所有出现次数超过最小次数的特征计算 ID 空间。通过将特征值 $\mathbf{x}$ 映射到其累积分布函数 $P(x≤x)$，将连续实值特征归一化为 $[0,1]$ ，并将其分为 $n_q$ 分位数。对于 $i$ -th 分位数的值，归一化值为 $\frac{i−1}{n_q - 1}$。分位数边界在数据生成期间计算。

在这个阶段，我们使用一段时间内的 user 和 app impression 数据来生成训练数据。每个例子对应一个 impression。标签是应用是否被获取：如果应用被安装，则为 1，否则为 0。

词汇表是将类别特征映射到 id 的表，也在这个阶段生成。系统为所有出现次数超过最小次数的特征计算 ID 空间。通过将特征值 $\mathbf{x}$ 映射到其累积分布函数 $P(x≤x)$，将连续实值特征归一化为 $[0,1]$ ，并将其分为 $n_q$ 分位数。对于 $i$ -th分位数的值，归一化值为 $\frac{i−1}{n_q - 1}$。分位数边界在数据生成期间计算。

#### 4.2 Model Training

我们在实验中使用的模型结构如图4所示。在训练过程中，我们的输入层接收训练数据和词汇表，并生成稀疏和密集的特征以及标签。wide 部分包括用户安装的应用程序和 impression 应用程序的 cross-product 变换。对于模型的 deep 部分，为每个类别特征学习一个 $32$ 维的 embedding 向量。

我们将所有的 embedding 和 dense 特征连接在一起，产生大约 $1200$ 维的 dense 向量。然后，将连接后的向量送入 3 个 RELU 层，最后送入逻辑输出单元(logistic output)。

Wide&Deep 模型对超过 5000亿 个例子进行了训练。每当一组新的训练数据到达时，模型都需要重新训练。然而，每次从头开始训练的计算成本很高，并且会延长从数据到达到提供更新好后的模型所需的时间。为了应对这一挑战，我们实现了一个热启动系统，该系统使用来自前一个模型的 embedding 和线性模型权重来初始化一个新模型。

在将模型加载到模型服务器之前，会对模型进行模拟运行，以确保在服务实时流量时不会出现问题。我们根据先前的模型对模型质量进行经验性的验证，以此作为一种合理性的检查。

![Figure4](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig4.png)

**Figure 4: Wide & Deep model structure for apps recommendation.**

#### 4.3 Model Serving

一旦模型经过训练和验证，我们就将其加载到模型服务器中。对于每个请求，服务器从应用程序检索系统接收一组候选应用程序和用户特征，以对每个应用程序进行打分。然后，应用程序从最高分到最低分进行排序，我们按照这个顺序向用户展示这些应用程序。分数是通过在 Wide&Deep 模型上运行正向推理过程来计算的。

为了在 10ms 量级上为每个请求提供服务，我们通过并行运行较小的批处理来使用多线程并行来优化性能，而不是在单个批处理推理步骤中对所有候选应用程序进行评分。

## 5. EXPERIMENT RESULTS

为了评估 Wide&Deep 学习在推荐系统中的实践有效性，我们进行了线上实验，并从应用程序获取和服务性能两个方面对系统进行了评估。

#### 5.1 App Acquisitions

我们在 A/B testing 框架中进行了为期 3 周的线上实验。对于控制组，随机选择 1% 的用户，并向他们展示由前一版本的排序模型生成的推荐列表，该模型是一种高度优化的 广义 Logistic regression 模型，具有丰富的 cross-product 特征变换。对于实验组，1% 的用户被呈现给由 Wide & Deep 模型生成的推荐，并使用相同的特征集进行训练。

如 表1 所示，Wide & Deep model 与对照组相比，应用商店主登陆页面的 APP 获取率提升了 +3.9%(统计显著)。结果还与仅使用具有相同特征和神经网络结构的模型的 deep 的另一个 1% 组进行了比较，Wide&Deep 模式在仅使用 deep 模型的基础上有 +1% 的增益(统计学意义)。

除了线上实验外，我们还给出了离线情况下 ROC 曲线下的面积(AUC)。虽然 Wide&Deep 的线下 AUC 略高，但对线上流量的影响更大。一个可能的原因是离线数据集中的 impression 和 label 是固定的，而 online 系统可以通过混合泛化性和记忆性来生成新的探索性推荐，并从新用户的 responses 中学习。

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Table1.png)

**Table 1: Offline & online metrics of different models. Online Acquisition Gain is relative to the control.**

#### 5.2 Serving Performance

我们的移动应用商店面临着高流量的情况下，提供高吞吐量和低延迟的服务是具有挑战性的。在流量高峰期，我们的推荐服务器每秒可获得超过 1000万 个应用程序的评分。

使用单线程，对一个 batch 中的候选集进行评分需要 31 毫秒。我们实现了多线程，并将每个批处理拆分为更小的大小，这显著地将客户端延迟降低到了 14ms (包括服务开销)，如表2所示。

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Table2.png)

**Table 2: Serving latency vs. batch size and threads.**

## 6. RELATED WORK

将 wide 线性模型与 cross-product 变换和深度神经网络与 dense embedding 相结合的想法受到了以前工作的启发，例如因子分解机[5]，它通过将两个变量之间的交互分解为两个 low-dimensional embedding 之间的点积来增加线性模型的泛化能力。在本文中，我们通过神经网络而不是点积来学习 embedding 之间的高度非线性相交互，从而加强了模型的能力。

在语言模型中，循环神经网络(RNN) 和具有 n-gram 特征的最大熵模型的联合训练被提出通过学习输入和输出之间的直接权重来显著降低 RNN 的复杂度(例如，隐藏层大小)[4]。在计算机视觉中，深度残差学习[2]已被用来降低训练更深层模型的难度，并通过跳过一层或多层的快捷连接来提高精度。神经网络和图形模型的联合训练也被应用于从图像中估计人体姿势[6]。在这项工作中，我们探索了前馈神经网络和线性模型的联合训练，稀疏特征和输出单元之间有直接的联系，用于稀疏输入数据的通用推荐和排序问题。

在推荐系统文献中，通过将内容信息的深度学习和评分矩阵的协同过滤(CF)结合起来，已经探索了 collaborativate deep learning[7]。之前也有关于移动应用推荐系统的工作，比如 AppJoy 在用户的应用使用记录中使用 CF[8]。与以往基于 CF 或基于内容的方法不同，我们针对APP推荐系统的用户和 impression 数据联合训练 Wide&Deep 模型。

## 7. CONCLUSION

对于推荐系统来说，记忆性和泛化性都是非常重要的。wide 线性模型可以通过 cross-product 特征变换有效地记忆稀疏特征交互，而深度神经网络可以通过 low dimensional embeddings 将稀疏特征交互推广到以前未曾见过的特征交互作用。为了结合这两种模型的优点，我们提出了 Wide&Deep 学习框架。我们在大型应用商店 Google Play 的推荐系统上对该框架进行了生产和评估。线上实验结果表明，Wide&Deep 模式比仅 Wide-Only 模式和 Depth-Only 模式在 APP 获取方面有显著改善。

## 8. REFERENCES

[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, July 2011. 

[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2016. 

[3] H. B. McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and l1 regularization. In Proc. AISTATS, 2011. 

[4] T. Mikolov, A. Deoras, D. Povey, L. Burget, and J. H. Cernocky. Strategies for training large scale neural network language models. In IEEE Automatic Speech Recognition & Understanding Workshop, 2011. 

[5] S. Rendle. Factorization machines with libFM. ACM Trans. Intell. Syst. Technol., 3(3):57:1–57:22, May 2012. 

[6] J. J. Tompson, A. Jain, Y. LeCun, and C. Bregler. Joint training of a convolutional network and a graphical model for human pose estimation. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, NIPS, pages 1799–1807. 2014. 

[7] H. Wang, N. Wang, and D.-Y. Yeung. Collaborative deep learning for recommender systems. In Proc. KDD, pages 1235–1244, 2015. 

[8] B. Yan and G. Chen. AppJoy: Personalized mobile application discovery. In MobiSys, pages 113–126, 2011.



--------------------

## QA

1. embedding 在排序模型中的作用

   > Abstract

2. 推荐系统中的记忆性和泛化性是什么

   > 1

3. 推荐系统中如何实现记忆性和泛化性

   > 1

4. 推荐系统中记忆性与泛化性在业务产品目标中对应是什么

   > 1

5. Wide&Deep模型结构

   > 1

6. Wide&Deep原理推导

   > 3

7. Wide&Deep在落地场景中的模型架构

   > 4

8. 线上服务的一些技巧

   > 5

9. Wide&Deep与FM的区别

   > 6

10. Wide&Deep如何实现泛化性和记忆性的

    > 7 