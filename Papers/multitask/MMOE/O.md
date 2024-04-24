# Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

## ABSTRACT

Neural-based multi-task learning has been successfully used in many real-world large-scale applications such as recommendation systems. For example, in movie recommendations, beyond providing users movies which they tend to purchase and watch, the system might also optimize for users liking the movies afterwards. With multi-task learning, we aim to build a single model that learns these multiple goals and tasks simultaneously. However, the prediction quality of commonly used multi-task models is often sensitive to the relationships between tasks. It is therefore important to study the modeling tradeos between task-specic objectives and inter-task relationships.

> 基于神经网络的多任务学习已成功应用于许多现实世界的大规模应用，如推荐系统。例如，在电影推荐中，除了为用户提供他们可能购买和观看的电影外，系统还可以优化用户观影后的喜好。通过多任务学习，我们的目标是建立一个模型，可以同时学习多个目标和任务。然而，常用多任务模型的预测质量通常对各任务之间的关系很敏感。因此，研究任务特定目标和任务间关系的建模权衡非常重要。

In this work, we propose a novel multi-task learning approach, Multi-gate Mixture-of-Experts (MMoE), which explicitly learns to model task relationships from data. We adapt the Mixture-ofExperts (MoE) structure to multi-task learning by sharing the expert submodels across all tasks, while also having a gating network trained to optimize each task. To validate our approach on data with dierent levels of task relatedness, we rst apply it to a synthetic dataset where we control the task relatedness. We show that the proposed approach performs better than baseline methods when the tasks are less related. We also show that the MMoE structure results in an additional trainability benet, depending on dierent levels of randomness in the training data and model initialization. Furthermore, we demonstrate the performance improvements by MMoE on real tasks including a binary classication benchmark, and a large-scale content recommendation system at Google.

> 在这项工作中，我们提出了一种新的多任务学习方法，即多门混合专家模型（MMoE），该模型可以从数据中明确学习任务之间的关系。我们通过跨所有任务共享专家子模型，将混合专家（MoE）结构应用于多任务学习，同时训练一个门控网络来优化每个任务。为了在不同任务相关性的数据上验证我们的方法，我们首先将其应用于我们可以控制任务相关性的合成数据集。我们表明，当任务相关性较低时，该方法的表现优于基线方法。我们还表明，MMoE结构带来了额外的可训练性优势，这取决于训练数据和模型初始化的不同随机级别。此外，我们展示了MMoE在真实任务上的性能提升，包括二元分类基准测试和谷歌的大规模内容推荐系统。

### 1 INTRODUCTION

In recent years, deep neural network models have been successfully applied in many real world large-scale applications, such as recommendation systems [11]. Such recommendation systems often need to optimize multiple objectives at the same time. For example, when recommending movies for users to watch, we may want the users to not only purchase and watch the movies, but also to like the movies afterwards so that they’ll come back for more movies. That is, we can create models to predict both users’ purchases and their ratings simultaneously. Indeed, many large-scale recommendation systems have adopted multi-task learning using Deep Neural Network (DNN) models [3].

> 近年来，深度神经网络模型已成功应用于许多现实世界的大规模应用中，如推荐系统[11]。这类推荐系统往往需要同时优化多个目标。例如，在为用户推荐电影观看时，我们可能不仅希望用户购买和观看电影，还希望用户在观影后能喜欢这部电影，以便他们回来观看更多电影。也就是说，我们可以创建模型来同时预测用户的购买行为和评分。事实上，许多大规模的推荐系统已经采用了基于深度神经网络（DNN）模型的多任务学习[3]。

Researchers have reported multi-task learning models can improve model predictions on all tasks by utilizing regularization and transfer learning [8]. However, in practice, multi-task learning models do not always outperform the corresponding single-task models on all tasks [23, 26]. In fact, many DNN-based multi-task learning models are sensitive to factors such as the data distribution dierences and relationships among tasks [15, 34]. The inherent conicts from task dierences can actually harm the predictions of at least some of the tasks, particularly when model parameters are extensively shared among all tasks.

> 研究人员报道，多任务学习模型可以利用正则化和迁移学习来提高所有任务的模型预测[8]。然而，在实践中，多任务学习模型并不总是在所有任务上都优于相应的单任务模型[23, 26]。事实上，许多基于DNN的多任务学习模型对任务间的数据分布差异和关系等因素很敏感[15, 34]。任务差异带来的固有冲突实际上可能会损害至少部分任务的预测，特别是当模型参数在所有任务中广泛共享时。

Prior works [4, 6, 8] investigated task dierences in multi-task learning by assuming particular data generation processes for each task, measuring task differences according to the assumption, and then making suggestions based on how different the tasks are. However, as real applications often have much more complicated data patterns, it is often diffcult to measure task dierences and to make use of the suggested approaches of these prior works.

> 先前的工作[4, 6, 8]通过假设每个任务有特定的数据生成过程来研究多任务学习中的任务差异，根据假设衡量任务差异，然后根据任务的差异程度提出建议。然而，由于实际应用中往往有更复杂的数据模式，因此很难衡量任务差异，也很难利用先前工作中建议的方法。

Several recent works proposed novel modeling techniques to handle task differences in multi-task learning without relying on an explicit task difference measurement [15, 27, 34]. However, these techniques often involve adding many more model parameters per task to accommodate task dierences. As large-scale recommendation systems can contain millions or billions of parameters, those additional parameters are often under-constrained, which may hurt model quality. The additional computational cost of these parameters are also often prohibitive in real production settings due to limited serving resource.

> 最近有几项工作提出了新颖的建模技术来处理多任务学习中的任务差异，而不依赖于明确的任务差异测量[15, 27, 34]。然而，这些技术通常涉及为每个任务增加更多的模型参数以适应任务差异。由于大规模的推荐系统可能包含数百万或数十亿个参数，这些额外的参数往往受到的限制较少，这可能会损害模型质量。在实际生产环境中，由于服务资源有限，这些参数的额外计算成本也往往是难以承担的。

In this paper, we propose a multi-task learning approach based on a novel Multi-gate Mixture-of-Experts (MMoE) structure, which is inspired by the Mixture-of-Experts (MoE) model [21] and the recent MoE layer [16, 31]. MMoE explicitly models the task relationships and learns task-specic functionalities to leverage shared representations. It allows parameters to be automatically allocated to capture either shared task information or task-specic information, avoiding the need of adding many new parameters per task.

> 在本文中，我们提出了一种基于新型多门混合专家（MMoE）结构的多任务学习方法，该方法受混合专家（MoE）模型[21]和最近的MoE层[16, 31]的启发。MMoE明确建模任务关系，并学习任务特定的功能以利用共享表示。它允许参数自动分配以捕获共享任务信息或任务特定信息，从而避免每个任务需要添加许多新参数。

The backbone of MMoE is built upon the most commonly used Shared-Bottom multi-task DNN structure [8]. The Shared-Bottom model structure is shown in Figure 1 (a), where several bottom layers following the input layer are shared across all the tasks and then each task has an individual “tower” of network on top of the bottom representations. Instead of having one bottom network shared by all tasks, our model, shown in Figure 1 (c), has a group of bottom networks, each of which is called an expert. In our paper, each expert is a feed-forward network. We then introduce a gating network for each task. The gating networks take the input features and output softmax , gates assembling the experts with different weights, allowing different tasks to utilize experts differently. The results of the assembled experts are then passed into the task-specic tower networks. In this way, the gating networks for different tasks can learn different mixture patterns of experts assembling, and thus capture the task relationships.

> MMoE的核心是建立在最常用的share-bottom多任务深度神经网络（DNN）结构上的[8]。share-bottom模型结构如图1（a）所示，其中输入层之后的几个底层在所有任务之间是共享的，然后每个任务在底部表示的基础上都有一个单独的“塔”式网络。与让所有任务共享一个底层网络不同，我们的模型（如图1（c）所示）有一组底层网络，每个底层网络都被称为一个专家。在我们的论文中，每个专家都是一个前馈网络。然后，我们为每个任务引入了一个门控网络。门控网络接收输入特征，并输出softmax，以不同的权重组合专家网络，允许不同的任务以不同的方式利用专家。组合专家的结果随后被传递到特定任务的“塔”式网络中。通过这种方式，不同任务的门控网络可以学习不同的专家组合模式，从而捕获任务之间的关系。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Fig1.png)

To understand how MMoE learns its experts and task gating networks for different levels of task relatedness, we conduct a synthetic experiment where we can measure and control task relatedness by their Pearson correlation. Similar to [24], we use two synthetic regression tasks and use sinusoidal functions as the data generation mechanism to introduce non-linearity. Our approach outperforms baseline methods under this setup, especially when task correlation is low. In this set of experiments, we also discover that MMoE is easier to train and converges to a better loss during multiple runs. This relates to recent discoveries that modulation and gating mechanisms can improve the trainability in training non-convex deep neural networks [10, 19].

> 为了理解MMoE如何为不同程度的任务相关性学习其专家和任务门控网络，我们进行了一个合成实验，在这个实验中，我们可以通过皮尔逊相关性来测量和控制任务相关性。与[24]类似，我们使用了两个合成回归任务，并使用正弦函数作为数据生成机制来引入非线性。在这种设置下，我们的方法优于基线方法，尤其是在任务相关性较低的情况下。在这组实验中，我们还发现MMoE更容易训练，并且在多次运行中收敛到更好的损失。这与最近的发现有关，即调制和门控机制可以改善非凸深度神经网络训练的可训练性[10, 19]。

We further evaluate the performance of MMoE on a benchmark dataset, UCI Census-income dataset, with a multi-task problem setup. We compare with several state-of-the-art multi-task models which model task relations with soft parameter sharing, and observe improvement in our method.

> 我们进一步在多任务问题设置下，使用基准数据集UCI Census-income来评估MMoE的性能。我们与几种使用软参数共享对任务关系进行建模的最先进的多任务模型进行了比较，并观察到我们的方法有所改进。

Finally, we test MMoE on a real large-scale content recommendation system, where two classication tasks are learned at the same time when recommending items to users. We train MMoE model with hundreds of billions of training examples and compare it with a shared-bottom production model. We observe signicant improvements in oine metrics such as AUC. In addition, our MMoE model consistently improves online metrics in live experiments.

> 最后，我们在一个真实的大规模内容推荐系统上测试MMoE，在向用户推荐项目时，会同时学习两个分类任务。我们用数百亿个训练样本来训练MMoE模型，并将其与共享底部的生产模型进行比较。我们观察到离线指标（如AUC）有显著提高。此外，我们的MMoE模型在在线实时实验中始终提高了在线指标。

The contribution of this paper is threefold: First, we propose a novel Multi-gate Mixture-of-Experts model which explicitly models task relationships. Through modulation and gating networks, our model automatically adjusts parameterization between modeling shared information and modeling task-specic information. Second, we conduct control experiments on synthetic data. We report how task relatedness affects training dynamics in multi-task learning and how MMoE improves both model expressiveness and trainability. Finally, we conduct experiments on real benchmark data and a large-scale production recommendation system with hundreds of millions of users and items. Our experiments verify the effciency and effectiveness of our proposed method in real-world settings.

> 本文的贡献有三方面：首先，我们提出了一种新型的多门混合专家模型，该模型可以明确地模拟任务关系。通过建模和门控网络，我们的模型可以自动调整建模共享信息和建模特定任务信息之间的参数化。其次，我们在合成数据上进行了对照实验。我们报告了任务相关性如何影响多任务学习中的训练动态，以及MMoE如何提高模型的表达能力和可训练性。最后，我们在真实的基准数据和拥有数亿用户和项目的大规模生产推荐系统上进行了实验。我们的实验验证了所提出的方法在现实环境中的效率和有效性。

## 2 RELATED WORK

#### 2.1 Multi-task Learning in DNNs

Multi-task models can learn commonalities and differences across different tasks. Doing so can result in both improved effciency and model quality for each task [4, 8, 30]. One of the widely used multi-task learning models is proposed by Caruana [8, 9], which has a shared-bottom model structure, where the bottom hidden layers are shared across tasks. This structure substantially reduces the risk of overfitting, but can suffer from optimization conflicts caused by task differences, because all tasks need to use the same set of parameters on shared-bottom layers.

> 多任务模型可以学习不同任务之间的共性和差异。这样做可以提高每个任务的效率和模型质量[4, 8, 30]。其中一种广泛使用的多任务学习模型是由Caruana提出的[8, 9]，该模型具有共享底部的模型结构，其中底部的隐藏层在任务之间是共享的。这种结构大大降低了过拟合的风险，但可能会受到由任务差异引起的优化冲突的影响，因为所有任务都需要在共享底层上使用同一组参数。

To understand how task relatedness affects model quality, prior works used synthetic data generation and manipulated different types of task relatedness so as to evaluate the effectiveness of multitask models [4–6, 8].

> 为了理解任务相关性如何影响模型质量，之前的工作使用了合成数据生成，并控制不同类型的任务相关性，以评估多任务模型的有效性[4-6, 8]。

Instead of sharing hidden layers and same model parameters across tasks, some recent approaches add different types of constraints on task-specic parameters [15, 27, 34]. For example, for two tasks, Duong et al. [15] adds L2 constraints between the two sets of parameters. The cross-stitch network [27] learns a unique combination of task-specic hidden-layer embeddings for each task. Yang et al. [34] uses a tensor factorization model to generate hidden-layer parameters for each task. Compared to shared-bottom models, these approaches have more task-specic parameters and can achieve better performance when task differences lead to conflicts in updating shared parameters. However, the larger number of task-specic parameters require more training data to t and may not be effcient in large-scale models

> 最近的一些方法不是跨任务共享隐藏层和相同的模型参数，而是对特定任务的参数添加了不同类型的约束[15, 27, 34]。例如，对于两个任务，Duong等人[15]在两组参数之间添加了L2约束。cross-stich 网络[27]为每个任务学习特定任务隐藏层embedding的独特组合。Yang等人[34]使用张量分解模型为每个任务生成隐藏层参数。与共享底部模型相比，这些方法具有更多的特定任务参数，并且当任务差异导致共享参数更新冲突时，可以实现更好的性能。然而，大量的特定任务参数需要更多的训练数据来适应，并且在大规模模型中可能并不高效。

#### 2.2 Ensemble of Subnets & Mixture of Experts

In this paper, we apply some recent findings in deep learning such as parameter modulation and ensemble method to model task relationships for multi-task learning. In DNNs, ensemble models and ensemble of subnetworks have been proven to be able to improve model performance [9, 20].

> 在本文中，我们将深度学习中的一些最新发现，如参数建模和集成方法，应用于多任务学习的任务关系建模。在深度神经网络中，已经证明集成模型和子网络集成能够提高模型性能[9, 20]。

Eigen et al [16] and Shazeer et al [31] turn the mixture-of-experts model into basic building blocks (MoE layer) and stack them in a DNN. The MoE layer selects subnets (experts) based on the input of the layer at both training time and serving time. Therefore, this model is not only more powerful in modeling but also lowers computation cost by introducing sparsity into the gating networks. Similarly, PathNet [17], which is designed for articial general intelligence to handle different tasks, is a huge neural network with multiple layers and multiple submodules within each layer. While training for one task, multiple pathways are randomly selected and trained by different workers in parallel. The parameters of the best pathway is fixed and new pathways are selected for training new tasks. We took inspiration from these works by using an ensemble of subnets (experts) to achieve transfer learning while saving computation.

> Eigen等人[16]和Shazeer等人[31]将混合专家模型转化为基本的构建模块（MoE层），并将其堆叠在深度神经网络中。MoE层根据层的输入在训练时和服务时选择子网（专家）。因此，该模型不仅在建模方面更强大，而且通过引入稀疏性到门控网络中降低了计算成本。类似地，PathNet[17]是为人工智能设计的，用于处理不同的任务，它是一个巨大的神经网络，具有多层和每层内的多个子模块。在训练一个任务时，会随机选择多条路径，并由不同的 worker 并行训练。最佳路径的参数是固定的，选择新的路径来训练新任务。我们从这些工作中汲取灵感，通过使用子网（专家）的集合来实现迁移学习，同时节省计算量。

#### 2.3 Multi-task Learning Applications

Thanks to the development of distributed machine learning systems [13], many large-scale real-world applications have adopted DNN-based multi-task learning algorithms and observed substantial quality improvements. On multi-lingual machine translation tasks, with shared model parameters, translation tasks having limited training data can be improved by jointly learning with tasks having large amount of training data [22]. For building recommendation systems, multi-task learning is found helpful for providing context-aware recommendations [28, 35]. In [3], a text recommendation task is improved by sharing feature representations and lower level hidden layers. In [11], a shared-bottom model is used to learn a ranking algorithm for video recommendation. Similar to these prior works, we evaluate our modeling approach on a realworld large-scale recommendation system. We demonstrate that our approach is indeed scalable, and has favorable performance compared with other state-of-the-art modeling approaches.

> 得益于分布式机器学习系统的发展[13]，许多大规模的实际应用采用了基于深度神经网络的多任务学习算法，并观察到了显著的质量提升。在多语言机器翻译任务中，通过共享模型参数，训练数据有限的翻译任务可以通过与具有大量训练数据的任务联合学习来提高效果[22]。在构建推荐系统时，多任务学习对于提供上下文感知推荐是有帮助的[28, 35]。在[3]中，通过共享特征表示和较低级别的隐藏层，提高了文本推荐任务的效果。在[11]中，使用一个共享底部模型来学习视频推荐的rank算法。与这些之前的工作类似，我们在一个真实世界的大规模推荐系统上评估了我们的建模方法。我们证明了我们的方法确实是可扩展的，并且与其他最先进的建模方法相比，具有更好的性能。

## 3 PRELIMINARY

#### 3.1 Shared-bottom Multi-task Model

We first introduce the shared-bottom multi-task model in Figure 1 (a), which is a framework proposed by Rich Caruana [8] and widely adopted in many multi-task learning applications [18, 29]. Therefore, we treat it as a representative baseline approach in multitask modeling.

> 我们首先在图1（a）中介绍共享底部的多任务模型，这是由Rich Caruana提出的框架[8]，并已被广泛用于许多多任务学习应用中[18, 29]。因此，我们将其视为多任务建模中的代表性基线方法。

Given $K$ tasks, the model consists of a shared-bottom network, represented as function $f$ , and $K$ tower networks $h^k$ , where $k = 1, 2, ...,K$ for each task respectively. The shared-bottom network follows the input layer, and the tower networks are built upon the output of the shared-bottom. Then individual output $y^k$ for each task follows the corresponding task-specic tower. For task $k$, the model can be formulated as,
$$
y_k=h^k(f(x)) \text {. }
$$

> 给定 $K$ 个任务，模型由一个共享底部网络和 $K$ 个 塔组成。共享底部网络可以表示为一个函数 $f$ ，而对于每个任务，分别有一个塔 $h^k$，其中 $k = 1, 2, ..., K$。共享底部网络连接在输入层之后，而塔楼网络则建立在共享底部的输出之上。然后，每个任务的单独输出 $y^k$ 跟随相应的任务特定塔。对于任务 $k$ ，模型可以表示为：
>
> $$
> y^k = h^k(f(x)) \text{。}
> $$
>

#### 3.2 Synthetic Data Generation

Prior works [15, 27] indicate that the performance of multi-task learning models highly depends on the inherent task relatedness in the data. It is however diffcult to study directly how task relatedness affects multi-task models in real applications, since in real applications we cannot easily change the relatedness between tasks and observe the effect. Therefore to establish an empirical study for this relationship, we first use synthetic data where we can easily measure and control the task relatedness.

> 先前的研究表明，多任务学习模型的性能高度依赖于数据中的任务相关性。然而，在实际应用中，直接研究任务相关性如何影响多任务模型是困难的，因为在实际应用中，我们无法轻易地改变任务之间的相关性并观察其影响。因此，为了建立这种关系的实证研究，我们首先使用可以容易地测量和控制任务相关性的合成数据。

Inspired by Kang et al. [24], we generate two regression tasks and use the Pearson correlation of the labels of these two tasks as the quantitative indicator of task relationships. Since we focus on DNN models, instead of the linear functions used in [24], we set the regression model as a combination of sinusoidal functions as used in [33]. Specically, we generate the synthetic data as follows.

> 受Kang等人的启发，我们生成了两个回归任务，并使用这两个任务标签的皮尔逊相关系数作为任务关系的定量指标。由于我们关注的是深度神经网络（DNN）模型，因此我们没有使用[24]中的线性函数，而是将回归模型设置为[33]中使用的正弦函数的组合。具体来说，我们按照以下方式生成合成数据。

(1) Given the input feature dimension $d$, we generate two orthogonal unit vectors $u_1,u_2 \in \mathbb{R}^d , i.e.$,
$$
u_1^T u_2=0,\left\|u_1\right\|_2=1,\left\|u_2\right\|_2=1 \text {. }
$$
(2) Given a scale constant $c$ and a correlation score $−1 ≤ p ≤ 1$, generate two weight vectors $w_1,w_2$ such that
$$
w_1=c u_1, w_2=c\left(p u_1+\sqrt{\left(1-p^2\right)} u_2\right) .
$$
(3) Randomly sample an input data point $x \in \mathbb{R}^d$ with each of its element from $\mathcal{N} (0, 1).$

(4) Generate two labels $y_1$,$y_2$ for two regression tasks as follows
$$
\begin{aligned}
& y_1=w_1^T x+\sum_{i=1}^m \sin \left(\alpha_i w_1^T x+\beta_i\right)+\epsilon_1 \\
& y_2=w_2^T x+\sum_{i=1}^m \sin \left(\alpha_i w_2^T x+\beta_i\right)+\epsilon_2
\end{aligned}
$$
where $\alpha_i , \beta_i ,i = 1, 2, ...,m$ are given parameters that control the shape of the sinusoidal functions and $
\epsilon_1, \epsilon_2 \underset{\sim}{\text { i.d }} \mathcal{N}(0,0.01)
$,

(5) Repeat (3) and (4) until enough data are generated.

> (1) 给定输入特征维度 $d$，我们生成两个正交单位向量 $u_1, u_2 \in \mathbb{R}^d$，即满足以下条件：
>
> $$
> u_1^T u_2=0,\left\|u_1\right\|_2=1,\left\|u_2\right\|_2=1 \text {. }
> $$
>
> (2) 给定一个比例常数 $c$ 和一个相关系数 $−1 ≤ p ≤ 1$，生成两个权重向量 $w_1, w_2$，使得
>
> $$
> w_1=c u_1, \quad w_2=c\left(p u_1+\sqrt{1-p^2} u_2\right) \text {. }
> $$
>
> (3) 随机采样一个输入数据点 $x \in \mathbb{R}^d$，其中每个元素都来自正态分布 $\mathcal{N} (0, 1)$。
>
> (4) 为两个回归任务生成两个标签 $y_1, y_2$，如下所示：
>
> $$
> \begin{aligned}
> & y_1=w_1^T x+\sum_{i=1}^m \sin \left(\alpha_i w_1^T x+\beta_i\right)+\epsilon_1 \\
> & y_2=w_2^T x+\sum_{i=1}^m \sin \left(\alpha_i w_2^T x+\beta_i\right)+\epsilon_2
> \end{aligned}
> $$
>
> 其中，$\alpha_i, \beta_i, i = 1, 2, ..., m$ 是给定的参数，用于控制正弦函数的形状，并且 $\epsilon_1, \epsilon_2$ 是独立同分布的随机噪声，服从 $\mathcal{N}(0, 0.01)$。
>
> (5) 重复步骤 (3) 和 (4)，直到生成足够的数据。
>
> 注意：在原文的公式中，$w_2$ 的表达式中的根号应包含整个 $1-p^2$，以确保 $w_2$ 的模长正确，并且当 $|p|=1$ 时，公式仍然有效。因此，在翻译中已对此进行了更正。同时，我也稍微调整了公式的排版和表述，以使其更符合中文的表达习惯。

Due to the non-linear data generation procedure, it’s not straightforward to generate tasks with a given label Pearson correlation. Instead, we manipulate the cosine similarity of the weight vectors in Eq 2, which is $cos(w_1,w_2) = p$, and measuring the resulting label Pearson correlation afterwards. Note that in the linear case where
$$
\begin{equation}
\begin{aligned}
& y_1=w_1^T x+\epsilon_1 \\
& y_2=w_2^T x+\epsilon_2,
\end{aligned}
\end{equation}
$$
the label Pearson correlation of $y_1,y_2$ is exactly $p$. In the nonlinear case, $y_1$ and $y_2$ in Eq 3 and Eq 4 are also positively correlated, as shown in Figure 2. In the rest of this paper, for simplicity, we refer to cosine similarity of the weight vectors as “task correlation”.

> 由于数据生成过程的非线性，直接生成具有给定标签皮尔逊相关系数的任务并不简单。因此，我们通过操作公式2中的权重向量的余弦相似度（即$cos(w_1, w_2) = p$），并在之后测量实际得到的标签皮尔逊相关系数。值得注意的是，在线性情况下，即：
>
> $$
> \begin{equation}
> \begin{aligned}
> & y_1 = w_1^T x + \epsilon_1 \\
> & y_2 = w_2^T x + \epsilon_2,
> \end{aligned}
> \end{equation}
> $$
>
> 标签$y_1$和$y_2$的皮尔逊相关系数恰好为$p$。然而，在非线性情况下，如公式3和公式4中定义的$y_1$和$y_2$也是正相关的，如图2所示。在本文的其余部分中，为了简化，我们将权重向量的余弦相似度称为“任务相关性”。
>
> 这里提到的“任务相关性”是通过调整权重向量的余弦相似度来实现的，它影响了任务标签之间的相关性。虽然在线性情况下，皮尔逊相关系数与余弦相似度有直接的关系，但在非线性情况下，这种关系可能不是线性的，但仍然存在一定的正相关。因此，通过控制权重向量的余弦相似度，我们可以间接地调控任务之间的相关性，以研究多任务学习模型在不同任务相关性条件下的性能。这种方法为多任务学习的实验研究提供了一个有用的工具。

![Fig2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Fig2.png)

![Fig3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Fig3.png)

#### 3.3 Impact of Task Relatedness

To verify that low task relatedness hurts model quality in a baseline multi-task model setup, we conduct control experiments on the synthetic data as follows. 

(1) Given a list of task correlation scores, generate a synthetic dataset for each score;

(2) Train one Shared-Bottom multi-task model on each of these datasets respectively while controlling all the model and training hyper-parameters to remain the same;

(3) Repeat step (1) and (2) hundreds of times with datasets generated independently but control the list of task correlation scores and the hyper-parameters the same;

(4) Calculate the average performance of the models for each task correlation score.

Figure 3 shows the loss curves for different task correlations. As expected, the performance of the model trends down as the task correlation decreases. This trend is general for many different hyper-parameter settings. Here we only show an example of the control experiment results in Figure 3. In this example, each tower network is a single-layer neural network with 8 hidden units, and the shared bottom network is a single-layer network with size=16. The model is implemented using TensorFlow [1] and trained using Adam optimizer [25] with the default setting. Note that the two regression tasks are symmetric so it’s sucient to report the results on one task. This phenomenon validates our hypothesis that the traditional multi-task model is sensitive to the task relationships.

> 为了验证在基线多任务模型设置中任务相关性低会损害模型质量，我们在合成数据上进行了如下对照实验：
>
> （1）给定一系列任务相关性得分，为每个得分生成一个合成数据集；
>
> （2）在这些数据集上分别训练一个Shared-Bottom多任务模型，同时控制所有模型和训练超参数保持一致；
>
> （3）独立生成数据集，重复步骤（1）和（2）数百次，但控制任务相关性得分列表和超参数保持不变；
>
> （4）计算每个任务相关性得分的模型平均性能。
>
> 图3显示了不同任务相关性的损失曲线。正如预期的那样，随着任务相关性的降低，模型的性能呈下降趋势。这种趋势在许多不同的超参数设置中都是普遍的。这里我们仅在图3中展示了一个控制实验结果的示例。在这个示例中，每个塔式网络是一个具有8个隐藏单元的单层神经网络，而共享底层网络是一个大小为16的单层网络。该模型是使用TensorFlow[1]实现的，并使用默认设置的Adam优化器[25]进行训练。请注意，两个回归任务是对称的，因此报告一个任务的结果就足够了。这一现象验证了我们的假设，即传统的多任务模型对任务关系敏感。

## 4 MODELING APPROACHES

#### 4.1 Mixture-of-Experts

**The Original Mixture-of-Experts (MoE) Model** [21] can be formulated as:
$$
\begin{equation}
y=\sum_{i=1}^n g(x)_i f_i(x),
\end{equation}
$$
where $\sum_{i=1}^n g(x)_i=1$ and $g(x)_i$ , the $i$ th logit of the output of $g(x)$, indicates the probability for expert $f_i$​ .

Here, $i=1, \ldots, n$ are $n$ expert networks and $g$ represents a gating network that ensembles the results from all experts. More specically, the gating network $g$ produces a distribution over the $n$ experts based on the input, and the nal output is a weighted sum of the outputs of all experts.

> **原始混合专家（Mixture-of-Experts，简称MoE）模型** [21] 可以表述为：
>
> $$
> \begin{equation}
> 
> y=\sum_{i=1}^n g(x)_i f_i(x),
> 
> \end{equation}
> $$
>
> 其中，$\sum_{i=1}^n g(x)_i=1$，且 $g(x)_i$ 是 $g(x)$ 输出的第 $i$ 个逻辑值，表示专家 $f_i$ 的概率。
>
> 在这里，$i=1, \ldots, n$ 表示 $n$ 个专家网络，而 $g$ 代表一个门控网络，它集成了所有专家的结果。更具体地说，门控网络 $g$ 根据输入在 $n$ 个专家上产生一个分布，最终输出是所有专家输出的加权和。
>

**MoE Layer** : While MoE was first developed as an ensemble method of multiple individual models, Eigen et al [16] and Shazeer et al [31] turn it into basic building blocks (MoE layer) and stack them in a DNN. The MoE layer has the same structure as the MoE model but accepts the output of the previous layer as input and outputs to a successive layer. The whole model is then trained in an end-to-end way.

> **MoE层**：虽然MoE最初是作为多个单独模型的集成方法开发的，但Eigen等人[16]和Shazeer等人[31]将其转化为基本的构建块（MoE层）并在深度神经网络（DNN）中堆叠它们。MoE层具有与MoE模型相同的结构，但接受前一层的输出作为输入，并输出到后续层。然后，整个模型以端到端的方式进行训练。

The main goal of the MoE layer structure proposed by Eigen et al [16] and Shazeer et al [31] is to achieve conditional computation [7, 12], where only parts of a network are active on a per-example basis. For each input example, the model is able to select only a subset of experts by the gating network conditioned on the input.

> Eigen等人[16]和Shazeer等人[31]提出的MoE层结构的主要目标是实现条件计算[7, 12]，即每个示例中仅网络的部分区域处于活跃状态。对于每个输入示例，模型能够通过门控网络根据输入条件选择一部分专家。

#### 4.2 Multi-gate Mixture-of-Experts

We propose a new MoE model that is designed to capture the task differences without requiring signicantly more model parameters compared to the shared-bottom multi-task model. The new model is called Multi-gate Mixture-of-Experts (MMoE) model, where the key idea is to substitute the shared bottom network $f$ in Eq 1 with the MoE layer in Eq 5. More importantly, we add a separate gating network $g^k$ for each task $k$. More precisely, the output of task $k$ is
$$
y_k=h^k\left(f^k(x)\right) \text {, }
\\
\text { where } f^k(x)=\sum_{i=1}^n g^k(x)_i f_i(x) \text {. }
$$
See Figure 1 (c) for an illustration of the model structure.

> 我们提出了一种新的MoE模型，该模型旨在捕获任务之间的差异，同时与共享底层的多任务模型相比，不需要显著增加模型参数。新模型被称为多门混合专家（MMoE）模型，其关键思想是用 公式5 中的MoE层替换公式1 中的共享底层网络 $f$。更重要的是，我们为每个任务 $k$ 添加了一个单独的门控网络 $g^k$ 。更确切地说，任务 $k$ 的输出为
>
> $$
> y_k=h^k\left(f^k(x)\right) \text {, }
> 
> \\
> 
> \text { 其中 } f^k(x)=\sum_{i=1}^n g^k(x)_i f_i(x) \text {. }
> $$
>
> 请参阅图1（c）以了解模型结构的图示。
>
> （注：它通过引入多个门控网络和专家网络来更好地处理不同任务之间的差异。每个任务都有一个单独的门控网络来选择相关的专家网络，从而生成针对该任务的特定表示。这种方法旨在提高多任务学习的性能和灵活性。）
>

Our implementation consists of identical multilayer perceptrons with ReLU activations. The gating networks are simply linear transformations of the input with a softmax layer:
$$
g^k(x)=\operatorname{softmax}\left(W_{g k} x\right),
$$
where $W_{g k} \in \mathbb{R}^{n \times d}$ is a trainable matrix. $n$ is the number of experts and $d$ is the feature dimension.

> 我们的实现由具有ReLU激活的相同多层感知器组成。门控网络只是输入的线性变换，后面接一个softmax层：
>
> $$
> g^k(x)=\operatorname{softmax}\left(W_{g k} x\right),
> $$
>
> 其中$W_{g k} \in \mathbb{R}^{n \times d}$是一个可训练的矩阵。$n$是专家的数量，$d$是特征的维度。
>
> （注：该段是MMoE模型中门控网络的具体实现方式。门控网络通过对输入进行线性变换后应用softmax函数来得到每个专家网络的权重。这种设计允许模型根据输入动态地调整每个专家网络对输出结果的贡献。）

Each gating network can learn to “select” a subset of experts to use conditioned on the input example. This is desirable for a flexible parameter sharing in the multi-task learning situation. As a special case, if only one expert with the highest gate score is selected, each gating network actually linearly separates the input space into $n$ regions with each region corresponding to an expert. The MMoE is able to model the task relationships in a sophisticated way by deciding how the separations resulted by different gates overlap with each other. If the tasks are less related, then sharing experts will be penalized and the gating networks of these tasks will learn to utilize different experts instead. Compared to the Shared-Bottom model, the MMoE only has several additional gating networks, and the number of model parameters in the gating network is negligible. Therefore the whole model still enjoys the benfet of knowledge transfer in multi-task learning as much as possible. 

> 每个门控网络都可以学习, 根据输入样本“选择”要使用的一组专家。这对于多任务学习场景中的灵活参数共享是理想的。作为一种特殊情况，如果只选择门控得分最高的一个专家，则每个门控网络实际上将输入空间线性划分为 $n$​ 个区域，每个区域对应一个专家。MMoE 能够通过决定不同门控产生的分隔如何相互重叠，以复杂的方式对任务关系进行建模。如果任务之间的相关性较小，那么共享专家将会受到惩罚，并且这些任务的门控网络将学习利用不同的专家。与Shared-Bottom模型相比，MMoE只有几个额外的门控网络，并且门控网络中的模型参数数量可以忽略不计。因此，整个模型仍然可以尽可能地享受多任务学习中的知识迁移的好处。
>
> （注：进一步解释了MMoE模型如何通过学习选择专家来实现灵活的任务间参数共享，以及这种机制如何使模型能够在多任务学习中更好地进行知识迁移。同时，它也强调了与Shared-Bottom模型相比，MMoE在增加少量参数的情况下能够实现更复杂的任务关系建模。）

To understand how introducing separate gating network for each task can help the model learn task-specic information, we compare with a model structure with all tasks sharing one gate. We call it One-gate Mixture-of-Experts (OMoE) model. This is a direct adaption of the MoE layer to the Shared-Bottom multi-task model. See Figure 1 (b) for an illustration of the model structure.

> 为了理解为每个任务引入单独的门控网络如何帮助模型学习任务特定信息，我们与所有任务共享一个门的模型结构进行了比较。我们称之为单门混合专家（OMoE）模型。这是MoE层对共享底层多任务模型的直接改编。请参阅图1（b）以了解模型结构的图示。

## 5 MMOE ON SYNTHETIC DATA

In this section, we want to understand if the MMoE model can indeed better handle the situation where tasks are less related. Similar to Section 3.3, we conduct control experiments on the synthetic data to investigate this problem. We vary the task correlation of the synthetic data and observe how the behavior changes for different models. We also conduct a trainability analysis and show that MoE based models can be more easily trained compared to Shared-Bottom models.

> 在本节中，我们想要了解MMoE模型是否确实可以更好地处理任务相关性较小的情况。与3.3节类似，我们在合成数据上进行对照实验来研究这个问题。我们改变合成数据的任务相关性，并观察不同模型的行为如何变化。我们还进行了可训练性分析，并表明与共享底层模型相比，基于MoE的模型更容易训练。

#### 5.1 Performance on Data with Different Task Correlations

We repeat the experiments in section 3.3 for the proposed MMoE model and two baseline models: the Shared-Bottom model and the OMoE model.

> 我们针对所提出的MMoE模型和两个基线模型（共享底层模型和OMoE模型）重复了3.3节中的实验。

**Model Structures**. The input dimension is 100. Both MoE based models have 8 experts with each expert implemented as a singlelayer network. The size of the hidden layers in the expert network is 16. The tower networks are still single-layer networks with size=8. We note that the total number of model parameters in the shared experts and the towers is 100 × 16 × 8 + 16 × 8 × 2 = 13056. For the baseline Shared-Bottom model, we still set the tower network as a single-layer network with size=8. We set the single-layer shared bottom network with size 13056/(100 + 8 × 2) ≈ 113.

> **模型结构**。输入维度为100。两个基于MoE的模型都有8个专家，每个专家都实现为单层网络。专家网络中的隐藏层大小为16。塔式网络仍然是单层网络，大小为8。我们注意到，共享专家和塔中的模型参数总数为100 × 16 × 8 + 16 × 8 × 2 = 13056。对于基线共享底层模型，我们仍将塔式网络设置为单层网络，大小为8。我们将单层共享底层网络的大小设置为13056/(100 + 8 × 2) ≈ 113。

**Results**. All the models are trained with the Adam optimizer and the learning rate is grid searched from [0.0001, 0.001, 0.01]. For each model-correlation pair setting, we have 200 runs with independent random data generation and model initialization. The average results are shown in Figure 4. The observations are outlined as follows:

(1) For all models, the performance on the data with higher correlation is better than that on the data with lower correlation. 

(2) The gap between performances on data with different correlations of the MMoE model is much smaller than that of the OMoE model and the Shared-Bottom model. This trend is especially obvious when we compare the MMoE model with the OMoE model: in the extreme case where the two tasks are identical, there is almost no difference in performance between the MMoE model and the OMoE model; when the correlation between tasks decreases, however, there is an obvious degeneration of performance for the OMoE model while there is little inuence on the MMoE model. Therefore, it’s critical to have task-specic gates to model the task differences in the low relatedness case. 

(3) Both MoE models are better than the Shared-Bottom model in all scenarios in terms of average performance. This indicates that the MoE structure itself brings additional benets. Following this observation, we show in the next subsection that the MoE models have better trainability than the SharedBottom model.

> **结果**。所有模型都使用 Adam 优化器进行训练，学习率从[0.0001, 0.001, 0.01]中进行网格搜索。对于每个模型-相关性对设置，我们进行了200次运行，每次运行都进行了独立的数据生成和模型初始化。平均结果如图4所示。观察结果概述如下：
>
> （1）对于所有模型，相关性较高的数据上的性能优于相关性较低的数据上的性能。
>
> （2）MMoE模型在不同相关性数据上的性能差距远小于OMoE模型和共享底层模型。当我们将MMoE模型与OMoE模型进行比较时，这种趋势尤其明显：在两个任务完全相同的情况下，MMoE模型和OMoE模型之间的性能几乎没有差异；但是，当任务之间的相关性降低时，OMoE模型的性能明显下降，而对MMoE模型的影响很小。因此，在低相关性情况下，拥有针对特定任务的门控来建模任务差异至关重要。
>
> （3）在所有场景中，就平均性能而言，两个基于MoE的模型都优于共享底层模型。这表明MoE结构本身带来了额外的好处。根据这一观察，我们在下一小节中展示了MoE模型比共享底层模型具有更好的可训练性。
>
> 这段详细描述了实验结果及其分析。通过对比不同模型在不同任务相关性数据上的表现，得出了几个重要观察结果，包括MMoE模型在处理低相关性任务时的优越性、MoE结构带来的性能提升以及MoE模型相对于共享底层模型在可训练性方面的优势。

![Fig4](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Fig4.png)

#### 5.2 Trainability

For large neural network models, we care much about their trainability, i.e., how robust the model is within a range of hyper-parameter settings and model initializations.

> 对于大型神经网络模型，我们非常关心它们的可训练性，即模型在一系列超参数设置和模型初始化中的稳健性。

Recently, Collins et al [10] nd that some gated RNN models (like LSTM and GRU) we thought to perform better than the vanilla RNN are simply easier to train rather than having better model capacities. While we have demonstrated that MMoE can better handle the situation where tasks are less related, we also want to have a deeper understanding how it behaves in terms of trainability.

> 最近，Collins等人[10]发现，我们认为比标准RNN表现更好的一些门控RNN模型（如LSTM和GRU）实际上只是更容易训练，而不是具有更好的模型容量。虽然我们已经证明MMoE可以更好地处理任务相关性较小的情况，但我们还想更深入地了解它在可训练性方面的表现。

With our synthetic data, we can naturally investigate the robustness of our model against the randomness in the data and model initialization. We repeat the experiments under each setting multiple times. Each time the data are generated from the same distribution but different random seeds and the models are also initialized differently. We plot the histogram of the final loss values from repeated runs in Figure 5.

> 通过我们的合成数据，我们可以自然地研究模型对数据随机性和模型初始化的鲁棒性。我们在每种设置下多次重复实验。每次数据都是从相同的分布但不同的随机种子生成的，模型也进行了不同的初始化。我们在图5中绘制了重复运行的最终损失值的直方图。

There are three interesting observations from the histogram. First, in all task correlation settings, the performance variances of Shared-Bottom model are much larger than those of the MoE based model. This means that Shared-Bottom models in general have much more poor quality local minima than the MoE based models do. Second, while the performance variance of OMoE models is similarly robust as that of MMoE models when task correlation is 1, the robustness of the OMoE has an obvious drop when the task correlation decreases to 0.5. Note that the only difference between MMoE and OMoE is whether there is a multi-gate structure. This validates the usefulness of the multi-gate structure in resolving bad local minima caused by the conflict from task difference. Finally, it’s worth to observe that the lowest losses of all the three models are comparable. This is not surprising as neural networks are theoretically universal approximator. With enough model capacity, there should exist a “right” Shared-Bottom model that learns both tasks well. However, note that this is the distribution of 200 independent runs of experiments. And we suspect that for larger and more complicated model (e.g. when the shared bottom network is a recurrent neural network), the chance of getting the “right” model of the task relationship will be even lower. Therefore, explicitly modeling the task relationship is still desirable.

> 从直方图中，我们可以观察到三个有趣的现象。
>
> 首先，在所有任务相关性设置中，共享底层模型的性能方差远大于基于MoE（混合专家）的模型。这意味着，与基于MoE的模型相比，共享底层模型通常更容易陷入低质量的局部最小值。换句话说，共享底层模型在寻找全局最优解时面临更大的挑战，其性能更容易受到初始化和训练过程中的随机因素影响。
>
> 其次，当任务相关性为1时，OMoE（One-gate Mixture of Experts，单门混合专家）模型的性能方差与MMoE（Multi-gate Mixture of Experts，多门混合专家）模型相当稳健。然而，当任务相关性降低到0.5时，OMoE的稳健性明显下降。值得注意的是，MMoE和OMoE之间的唯一区别在于是否存在多门结构。这一观察结果验证了多门结构在解决由任务差异引起的冲突以及避免劣质局部最小值方面的有效性。多门结构能够更好地处理不同任务之间的复杂关系，从而提高模型的稳健性和性能。
>
> 最后，值得注意的是，这三个模型的最低损失是相当的。这并不奇怪，因为从理论上讲，神经网络是通用的逼近器，只要模型容量足够大，就应该存在一个能够同时学好两个任务的“正确”共享底层模型。然而，需要指出的是，这是基于200次独立实验运行的分布结果。我们怀疑，对于更大、更复杂的模型（例如，当共享底层网络是一个循环神经网络时），获得正确反映任务关系的模型的机会将会更低。因此，明确建模任务关系仍然是有意义的。通过显式地建模任务之间的关系，我们可以更好地理解和优化多任务学习中的复杂交互，从而提高模型的性能和泛化能力。

## 6 REAL DATA EXPERIMENTS

In this section, we conduct experiments on real datasets to validate the effectiveness of our approach.

> 在本节中，我们在真实数据集上进行实验，以验证我们方法的有效性。

#### 6.1 Baseline Methods

Besides the Shared-Bottom multi-task model, we compare our approach with several state-of-the-art multi-task deep neural network models that attempt to learn the task relationship from the data

> 除了共享底层多任务模型外，我们还将我们的方法与几种最先进的多任务深度神经网络模型进行了比较，这些模型试图从数据中学习任务关系。

**L2-Constrained** [15]: This method is designed for a cross-lingual problem with two tasks. In this method, parameters used for different tasks are shared softly by an L2 constraint.

Given $y_k$ as the ground truth label for task $k, k \in 1, 2$ , the prediction of task $k$ is represented as
$$
\hat{y}_k=f\left(x ; \theta_k\right),
$$
where $\theta_k$ are model parameters.

The objective function of this method is
$$
\mathbb{E} L\left(y_1, f\left(x ; \theta_1\right)\right)+\mathbb{E} L\left(y_2, f\left(x ; \theta_2\right)\right)+\alpha\left\|\theta_1-\theta_2\right\|_2^2
$$
where $y_1,y_2$ are the ground truth label for task 1 and task 2, and $\alpha$ is a hyper-parameter. This method models the task relatedness with the magnitude of $\alpha$.

> L2-Constrained 方法是为一个包含两个任务的 cross-lingual 问题设计的。在这种方法中，不同任务所用的参数通过一个 L2-Constrained 进行软共享。
>
> 给定 $y_k$ 作为任务 $k, k \in \{1, 2\}$ 的真实标签，任务 $k$ 的预测表示为：
>
> $$
> \hat{y}_k=f\left(x ; \theta_k\right),
> $$
> 其中 $\theta_k$ 是模型参数。
>
> 该方法的目标函数是：
>
> $$
> \mathbb{E} L\left(y_1, f\left(x ; \theta_1\right)\right)+\mathbb{E} L\left(y_2, f\left(x ; \theta_2\right)\right)+\alpha\left\|\theta_1-\theta_2\right\|_2^2,
> $$
> 其中 $y_1,y_2$ 分别是任务1和任务2的真实标签，$\alpha$ 是一个超参数。该方法使用 $\alpha$ 的大小来建模任务之间的相关性。
>
> 解释：
>
> - $f\left(x ; \theta_k\right)$：这是一个以 $x$ 为输入，以 $\theta_k$ 为参数的函数，用于预测任务 $k$ 的输出。
> - $\mathbb{E} L\left(y_k, f\left(x ; \theta_k\right)\right)$：这表示任务 $k$ 的预测损失与真实标签 $y_k$ 之间的期望损失。$L$ 是损失函数，例如均方误差或交叉熵损失等。
> - $\alpha\left\|\theta_1-\theta_2\right\|_2^2$：这是一个L2正则化项，用于约束两个任务的参数 $\theta_1$ 和 $\theta_2$ 之间的差异。$\alpha$ 是一个权重参数，控制这个正则化项的影响程度。当 $\alpha$ 较大时，模型会倾向于使两个任务的参数更加接近，从而加强任务之间的共享；当 $\alpha$ 较小时，模型对两个任务参数的差异更加容忍。
>
> 通过这种方式，L2-Constrained方法能够在两个任务之间找到一个平衡点，既允许任务之间有一定的差异，又通过共享参数来提高模型的泛化能力。这种方法在处理多任务学习时特别有用，尤其是当任务之间有一定相关性但又不完全相同时。

**Cross-Stitch** [27]: This method shares knowledge between two tasks by introducing a “Cross-Stitch” unit. The Cross-Stitch unit takes the input of separated hidden layers $x_1$ and $x_2$ from task 1 and 2, and outputs $\tilde{x}_1^i$ and $\tilde{x}_2^i$ respectively by the following equation:
$$
\left[\begin{array}{l}
\tilde{x}_1^i \\
\tilde{x}_2^i
\end{array}\right]=\left[\begin{array}{ll}
\alpha_{11} & \alpha_{12} \\
\alpha_{21} & \alpha_{22}
\end{array}\right]\left[\begin{array}{l}
x_1^i \\
x_2^i
\end{array}\right],
$$
where $\alpha_{jk}$, $k = 1, 2$ is a trainable parameter representing the cross transfer from task $k$ to task $j$. The $\tilde{x}_1^i$ and $\tilde{x}_2^i$ are sent to the higher level layer in task 1 and task 2 respectively.

> **Cross-Stitch**方法通过引入一个“Cross-Stitch”单元来在两个任务之间共享知识。Cross-Stitch单元从任务1和任务2中分别接收分离的隐藏层输入 $x_1$ 和 $x_2$，并通过以下方程分别输出 $\tilde{x}_1^i$ 和 $\tilde{x}_2^i$:
> $$
> \left[\begin{array}{l}
> \tilde{x}_1^i \\
> \tilde{x}_2^i
> \end{array}\right]=\left[\begin{array}{ll}
> \alpha_{11} & \alpha_{12} \\
> \alpha_{21} & \alpha_{22}
> \end{array}\right]\left[\begin{array}{l}
> x_1^i \\
> x_2^i
> \end{array}\right],
> $$
>
> 在这个方程中，$\alpha_{jk}$（$k = 1, 2$）是一个可训练的参数，表示从任务 $k$ 到任务 $j$ 的交叉传输。计算得到的 $\tilde{x}_1^i$ 和 $\tilde{x}_2^i$ 分别被发送到任务1和任务2的更高层级中。
>

**Tensor-Factorization** [34]: In this method, weights from multiple tasks are modeled as tensors and tensor factorization methods are used for parameter sharing across tasks. For our comparison, we implement Tucker decomposition for learning multi-task models, which is reported to deliver the most reliable results [34]. For example, given input hidden-layer size $m$, output hidden-layer size $n$ and task number $k$, the weights $\mathcal{W}$, which is a $m × n × k$ tensor, is derived from the following equation:
$$
\mathcal{W}=\sum_{i_1}^{r_1} \sum_{i_2}^{r_2} \sum_{i_3}^{r_3} S\left(i_1, i_2, i_3\right) \cdot U_1\left(:, i_1\right) \circ U_2\left(:, i_2\right) \circ U_3\left(:, i_3\right),
$$
where tensor  $S$ of size $r_1 × r_2 × r_3$, matrix $U_1$ of size $m × r_1$, $U_2$ of size $n × r_2$, and $U_3$ of size $k × r_3$ are trainable parameters. All of them are trained together via standard backpropagation. $r_1$, $r_2$ and $r_3$​ are hyper-parameters.

> **Tensor-Factorization**方法在多任务学习中，将多个任务的权重建模为张量，并使用张量分解方法进行任务间的参数共享。为了进行比较，我们实现了Tucker分解来学习多任务模型，据报道这种方法可以提供最可靠的结果。
>
> 具体来说，给定输入隐藏层大小 $m$、输出隐藏层大小 $n$ 和任务数量 $k$，权重 $\mathcal{W}$ 是一个 $m \times n \times k$ 的张量，它可以通过以下方程得出：
>
> $$\mathcal{W}=\sum_{i_1}^{r_1} \sum_{i_2}^{r_2} \sum_{i_3}^{r_3} S\left(i_1, i_2, i_3\right) \cdot U_1\left(:, i_1\right) \circ U_2\left(:, i_2\right) \circ U_3\left(:, i_3\right),$$
>
> 其中，张量 $S$ 的大小为 $r_1 \times r_2 \times r_3$，矩阵 $U_1$ 的大小为 $m \times r_1$，$U_2$ 的大小为 $n \times r_2$，$U_3$ 的大小为$k \times r_3$。这些都是可训练的参数，通过标准的反向传播算法一起进行训练。$r_1$、$r_2$ 和 $r_3$ 是超参数，用于控制分解的秩。
>
> 解释：
>
> - Tucker分解是一种高阶张量分解方法，它将一个张量分解为一个核心张量（在这里是$S$）和一系列因子矩阵（在这里是$U_1$、$U_2$和$U_3$）的外积。
> - 这种分解允许我们捕捉不同任务之间的共享结构和特定于任务的信息。核心张量$S$编码了任务之间的共享信息，而因子矩阵则允许模型捕获每个任务的独特性。
> - 通过调整超参数$r_1$、$r_2$和$r_3$，我们可以控制模型的复杂性和任务之间信息共享的程度。
> - 使用Tucker分解进行多任务学习的一个主要优点是它能够灵活地捕捉任务之间的相关性，同时保持模型的紧凑性。
>
> 在实际应用中，Tensor-Factorization方法可以通过标准的深度学习框架实现，并利用反向传播算法进行训练。这种方法在多任务学习场景中特别有用，尤其是当任务之间具有复杂的相关性结构时。



#### 6.2 Hyper-Parameter Tuning

We adopt a hyper-parameter tuner, which is used in recent deep learning frameworks [10], to search the best hyperparameters for all the models in the experiments with real datasets. The tuning algorithm is a Gaussian Process model similar to Spearmint as introduced in [14, 32].

> 我们采用了一种超参数调优器，这是最近在深度学习框架中使用的工具，用于在实际数据集的实验中为所有模型搜索最佳超参数。该调优算法是一个高斯过程模型，类似于[14, 32]中介绍的Spearmint。

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Figure5.png)

To make the comparison fair, we constrain the maximum model size of all methods by setting a same upper bound for the number of hidden units per layer, which is 2048. For MMoE, it is the “number of experts” × “hidden units per expert”. Our approach and all baseline methods are implemented using TensorFlow [1]. 

We tune the learning rates and the number of training steps for all methods. We also tune some method-specic hyper-parameters: 

- MMOE: Number of experts, number of hidden units per expert. 
- L2-Constrained: Hidden-layer size. Weight $\alpha$ of the L2 constraint. 
- Cross-Stitch: Hidden-layer size, Cross-Stitch layer size. 
- Tensor-Factorization: $r_1$, $r_2$, $r_3$ for Tuck Decomposition, hidden-layer size.

> 为了确保比较的公平性，我们通过为每层的隐藏单元数量设置一个相同的上限来限制所有方法的最大模型大小，该上限为2048。对于MMoE（Multi-gate Mixture-of-Experts）来说，这个限制应用于“专家数量”乘以“每个专家的隐藏单元数”。我们的方法和所有基线方法都是使用TensorFlow实现的。
>
> 我们为所有方法调整了学习率和训练步数。我们还调整了一些特定于方法的超参数：
>
> - MMOE：专家数量、每个专家的隐藏单元数。这些是MMoE特有的参数，用于控制模型中专家的数量和每个专家的复杂度。
> - L2-Constrained：隐藏层大小、L2约束的权重 $\alpha$。L2约束用于防止模型过拟合，通过调整隐藏层大小和L2权重来平衡模型的复杂度和正则化强度。
> - Cross-Stitch：隐藏层大小、Cross-Stitch层大小。Cross-Stitch方法允许不同任务之间共享信息，通过调整这些参数可以控制信息共享的程度和模型的复杂度。
> - Tensor-Factorization：Tucker分解的 $r_1$、$r_2$、$r_3$以及隐藏层大小。这些参数控制着张量分解的精度和模型的复杂度，通过调整它们可以找到最佳的模型性能。
>
> 通过对这些超参数的细致调整，我们可以确保每个方法都在相似的模型大小约束下进行比较，从而使得实验结果更加公正和可信。这样的实验设置有助于我们准确评估不同多任务学习方法在相同条件下的性能表现。

#### 6.3 Census-income Data

In this subsection, we report and discuss experiment results on the census-income data. 

**6.3.1 Dataset Description.** 

The UCI census-income dataset [2] is extracted from the 1994 census database. It contains 299,285 instances of demographic information of American adults. There are 40 features in total. We construct two multi-task learning problems from this dataset by setting some of the features as prediction targets and calculate the absolute value of Pearson correlation of the task labels over 10,000 random samples: 

1. Task 1: Predict whether the income exceeds $50K; 

   Task 2: Predict whether this person’s marital status is never married. Absolute Pearson correlation: 0.1768.

2. Task 1: Predict whether the education level is at least college; 

   Task 2: Predict whether this person’s marital status is never married. 

Absolute Pearson correlation: 0.2373.

In the dataset, there are 199,523 training examples and 99,762 test examples. We further randomly split test examples into a validation dataset and a test dataset by the fraction of 1:1.

Note that we remove education and marital status from input features as they are treated as labels in these setups. We compare MMoE with aforementioned baseline methods. Since both groups of tasks are binary classication problems, we use AUC scores as the evaluation metrics. In both groups, we treat the marital status task as the auxiliary task, and treat the income task in the first group and the education task in the second group as the main tasks. For hyper-parameter tuning, we use the AUC of the main task on the validation set as the objective. For each method, we use the hyper-parameter tuner conducting thousands of experiments to find the best hyper-parameter setup. After the hyper-parameter tuner finds the best hyper-parameter for each method, we train each method on training dataset 400 times with random parameter initialization and report the results on the test dataset.

> UCI census-income 数据集是从1994年的人口普查数据库中提取的。它包含了299,285个美国成年人的人口统计信息实例，总共有40个特征。我们通过将这个数据集中的一些特征设定为预测目标，从而构建了两个多任务学习问题，并计算了10,000个随机样本的任务标签的皮尔逊相关系数的绝对值：
>
> 1. 任务1：预测收入是否超过$50K；
>    任务2：预测这个人的婚姻状况是否为未婚。皮尔逊相关系数的绝对值为0.1768。
>
> 2. 任务1：预测教育程度是否至少为大学；
>    任务2：预测这个人的婚姻状况是否为未婚。
>    皮尔逊相关系数的绝对值为0.2373。
>
> 在数据集中，有199,523个训练样本和99,762个测试样本。我们进一步将测试样本按1:1的比例随机划分为验证数据集和测试数据集。
>
> 请注意，我们从输入特征中移除了教育和婚姻状况，因为它们在这些设置中被视为标签。我们将MMoE与上述基线方法进行了比较。由于这两组任务都是二元分类问题，因此我们使用AUC分数作为评估指标。在这两组中，我们都将婚姻状况任务视为辅助任务，将第一组中的收入任务和第二组中的教育任务视为主要任务。对于超参数调整，我们使用验证集上主要任务的AUC作为目标。对于每种方法，我们使用超参数调优器进行了数千次实验，以找到最佳的超参数设置。在超参数调优器为每种方法找到最佳超参数后，我们在训练数据集上对每个方法进行400次训练，每次训练都使用随机的参数初始化，并在测试数据集上报告结果。
>
> 这样的实验设置能够全面而严谨地评估不同多任务学习方法在相同条件下的性能，确保实验结果的公正性和可信度。通过多次训练和随机初始化参数，我们可以减少随机性对实验结果的影响，从而更准确地评估每种方法的性能。

**6.3.2 Results**. 

For both groups, we report the mean AUC over 400 runs, and the AUC of the run where best main task performance is obtained. Table 1 and Table 2 show the results of two groups of tasks. We also tune and train single-task models by training a separate model for each task and report their results.

Given the task relatedness (roughly measured by the Pearson correlation) is not very strong in either group, the Shared-Bottom model is almost always the worst among multi-task models (except for Tensor-Factorization). Both L2-Constrained and Cross-Stitch have separate model parameters for each task and add constraints on how to learn these parameters, and therefore perform better than Shared-Bottom. However, having constraints on model parameter learning heavily relies on the task relationship assumptions, which is less exible than the parameter modulation mechanism used by MMoE. So MMoE outperforms other multi-task models in all means in group 2, where the task relatedness is even smaller than group 1.

The Tensor-Factorization method is the worst in both groups. This is because it tends to generalize the hidden-layer weights for all of the tasks in lower rank tensor and matrices. This method can be very sensitive to task relatedness, since it tends to over-generalize when tasks are less related, and needs more data and longer time to train.

The multi-task models are not tuned for the auxiliary marital status task on validation set while the single-task model is. So it is reasonable that the single-task model gets the best performance on the auxiliary task.

> 对于这两组任务，我们报告了400次运行的平均AUC和获得最佳主要任务性能的运行的AUC。表1和表2显示了两组任务的结果。我们还通过为每个任务训练一个单独的模型来调整和训练单任务模型，并报告了它们的结果。
>
> 鉴于两组任务中的任务相关性（通过皮尔逊相关系数大致衡量）都不是非常强，因此Shared-Bottom模型在多任务模型中几乎总是最差的（Tensor-Factorization除外）。L2-Constrained和Cross-Stitch都为每个任务提供了单独的模型参数，并对如何学习这些参数增加了约束，因此它们的性能优于Shared-Bottom。然而，对模型参数学习的约束在很大程度上依赖于任务关系假设，这比MMoE使用的参数调制机制更不灵活。因此，在任务相关性甚至小于第一组的第二组中，MMoE在所有方面都优于其他多任务模型。
>
> Tensor-Factorization方法在两组中都是最差的。这是因为它倾向于将所有任务的隐藏层权重概括为较低阶的张量和矩阵。这种方法可能对任务相关性非常敏感，因为当任务相关性较小时，它倾向于过度概括，并且需要更多的数据和更长的时间来训练。
>
> 多任务模型并未针对验证集上的辅助婚姻状况任务进行调整，而单任务模型则进行了调整。因此，单任务模型在辅助任务上获得最佳性能是合理的。
>
> 总的来说，我们的实验结果表明，在任务相关性不是很强的情况下，MMoE由于其灵活的参数调制机制而表现出色。而其他方法在不同程度上依赖于任务之间的相关性，因此在任务相关性较弱时可能表现不佳。这也强调了选择合适的多任务学习方法时需要考虑任务之间的相关性。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Table1.png)

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Table2.png)

#### 6.4 Large-scale Content Recommendation

In this subsection, we conduct experiments on a large-scale content recommendation system in Google Inc., where the recommendations are generated from hundreds of millions of unique items for billions of users. Specically, given a user’s current behavior of consuming an item, this recommendation system targets at showing the user a list of relevant items to consume next.

Our recommendation system adopts similar framework as proposed in some existing content recommendation frameworks [11], which has a candidate generator followed by a deep ranking model. The deep ranking model in our setup is trained to optimize for two types of ranking objectives: (1) optimizing for engagement related objectives such as click through rate and engagement time; (2) optimizing for satisfaction related objectives, such as like rate. Our training data include hundreds of billions of user implicit feedbacks such as clicks and likes. If trained separately, the model for each task needs to learn billions of parameters. Therefore, compared to learning multiple objectives separately, a Shared-Bottom architecture comes with the benet of smaller model size. In fact, such a Shared-Bottom model is already used in production.

> 在本小节中，我们在谷歌公司的一个大规模内容推荐系统上进行了实验，该系统的推荐是从数亿个唯一项目中为数十亿用户生成的。具体来说，给定用户当前消费项目的行为，此推荐系统的目标是向用户展示一系列相关项目以供其接下来消费。
>
> 我们的推荐系统采用了与现有的一些内容推荐框架类似的框架，其中包括一个候选生成器和一个深度排序模型。在我们的设置中，深度排序模型被训练为优化两种类型的排序目标：（1）优化与参与度相关的目标，如点击率和参与时间；（2）优化与满意度相关的目标，例如点赞率。我们的训练数据包括数百亿用户的隐式反馈，如点击和点赞。如果单独训练，每个任务的模型都需要学习数十亿个参数。因此，与单独学习多个目标相比，Shared-Bottom架构具有模型尺寸更小的优势。事实上，这样一个Shared-Bottom模型已经在生产中使用了。

**6.4.1 Experiment Setup.** 

We evaluate the multi-task models by creating two binary classication tasks for the deep ranking model: (1) predicting a user engagement related behavior; (2) predicting a user satisfaction related behavior. We name these two tasks as engagement subtask and satisfaction subtask. 

Our recommendation system uses embeddings for sparse features and normalizes all dense features to [0, 1]scale. For the SharedBottom model, we implement the shared bottom network as a feedforward neural network with several fully-connected layers with ReLU activation. A fully-connected layer built on top of the shared bottom network for each task serves as the tower network. For MMoE, we simply change the top layer of the shared bottom network to an MMoE layer and keep the output hidden units with the same dimensionality. Therefore, we don’t add extra noticeable computation costs in model training and serving. We also implement baseline methods such as L2-Constrained and Cross-Stitch. Due to their model architectures, they have roughly double the number of parameters comparing to the Shared-Bottom model. We do not compare with Tensor-Factorization because the computation of the Tucker product cannot scale up to billion level without heavy eciency engineering. All models are optimized using mini-batch Stochastic Gradient Descent (SGD) with batch size 1024.

> 我们通过为深度排序模型创建两个二元分类任务来评估多任务模型：（1）预测用户参与度相关行为；（2）预测用户满意度相关行为。我们将这两个任务命名为参与度子任务和满意度子任务。
>
> 我们的推荐系统使用嵌入来处理稀疏特征，并将所有密集特征归一化到[0,1]范围。对于SharedBottom模型，我们将共享底层网络实现为一个具有几个带ReLU激活的全连接层的前馈神经网络。每个任务在共享底层网络之上构建的全连接层充当塔式网络。对于MMoE，我们只需将共享底层网络的顶层更改为MMoE层，并保持输出隐藏单元的维度相同。因此，我们不会在模型训练和服务中增加额外的明显计算成本。我们还实现了基线方法，如L2-Constrained和Cross-Stitch。由于它们的模型架构，与Shared-Bottom模型相比，它们的参数数量大约是前者的两倍。我们没有与Tensor-Factorization进行比较，因为如果没有大量的效率工程，Tucker产品的计算无法扩展到十亿级别。所有模型都使用批处理大小为1024的小批量随机梯度下降（SGD）进行优化。

**6.4.2 Oline Evaluation Results.**

For offline evaluation, we train the models on a fixed set of 30 billion user implicit feedbacks and evaluate on a 1 million hold-out dataset. Given that the label of the satisfaction subtask is much sparser than the engagement subtask, the online results have very high noise levels. We only show the AUC scores and R-Squared scores on the engagement subtask in Table 3.

We show the results after training 2 million steps (10 billion examples with batch size 1024), 4 million steps and 6 million steps. MMoE outperforms other models in terms of both metrics. L2- Constrained and Cross-Stitch are worse than the Shared-Bottom model. This is likely because these two models are built upon two separate single-task models and have too many model parameters to be well constrained.

To better understand how the gates work, we show the distribution of the softmax gate of each task in Figure 6. We can see that MMoE learns the difference between these two tasks and automatically balances the shared and non-shared parameters. Since satisfaction subtask’s labels are sparser than the engagement subtask’s, the gate for satisfaction subtask is more focused on a single expert.

> 为了进行离线评估，我们在一个包含300亿用户隐式反馈的固定数据集上训练模型，并在一个包含100万个样本的保留数据集上进行评估。鉴于满意度子任务的标签比参与度子任务的标签稀疏得多，因此在线结果具有很高的噪声水平。我们仅在表3中显示了参与度子任务的AUC分数和R平方分数。
>
> 我们展示了训练200万步（100亿个样本，批处理大小为1024）、400万步和600万步后的结果。MMoE在两项指标上都优于其他模型。L2-Constrained和Cross-Stitch比Shared-Bottom模型更差。这可能是因为这两个模型是建立在两个独立的单任务模型之上的，并且具有太多的模型参数而无法得到很好的约束。
>
> 为了更好地理解门的工作原理，我们在图6中展示了每个任务的softmax门的分布。我们可以看到，MMoE学习了这两个任务之间的差异，并自动平衡了共享和非共享参数。由于满意度子任务的标签比参与度子任务的标签更稀疏，因此满意度子任务的门更侧重于单个专家。
>
> 这段文字主要描述了离线评估的过程和结果，包括模型的训练数据集、评估数据集、评估指标以及不同模型的性能比较。同时，还通过可视化方式展示了MMoE模型中门的工作原理，以及不同任务之间门的分布差异。

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Fig6.png)

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Table3.png)

**6.4.3 Live Experiment Results**

At last, we conduct live experiments for our MMoE model on the content recommendation system. We do not conduct live experiments for L2-Constrained and CrossStitch methods because both models double the serving time by introducing more parameters.

We conduct two sets of experiments. The rst experiment is to compare a Shared-Bottom model with a Single-Task model. The Shared-Bottom model is trained on both engagement subtask and satisfaction subtask. The Single-Task model is trained on the engagement subtask only. Note that though not trained on the satisfaction subtask, the Single-Task model serves as a ranking model at test time so we can also calculate satisfaction metrics on it. The second experiment is to compare our MMoE model with the Shared-Bottom model in the rst experiment. Both experiments are done using the same amount of live trac.

Table 4 shows the results of these live experiments. First, by using Shared-Bottom model, we see a huge improvement on the satisfaction live metric of 19.72%, and a slight decrease of -0.22% on the engagement live metric. Second, by using MMoE, we improve both metrics comparing with the Shared-Bottom model. In this recommendation system, engagement metric has a much larger raw value than the satisfaction metric, and it is desirable to have no engagement metric loss or even gains while improving satisfaction metric.

> 最后，我们在内容推荐系统上对MMoE模型进行了在线实验。我们没有对L2-Constrained和CrossStitch方法进行在线实验，因为这两种模型通过引入更多参数使服务时间加倍。
>
> 我们进行了两组实验。第一组实验是比较Shared-Bottom模型和Single-Task模型。Shared-Bottom模型同时在参与度和满意度子任务上进行训练。而Single-Task模型仅在参与度子任务上进行训练。请注意，尽管没有在满意度子任务上进行训练，但Single-Task模型在测试时充当排名模型，因此我们也可以在其上计算满意度指标。第二组实验是比较第一组实验中我们的MMoE模型和Shared-Bottom模型。两组实验都使用了相同数量的实时跟踪数据。
>
> 表4显示了这些在线实验的结果。首先，通过使用Shared-Bottom模型，我们看到满意度实时指标大幅提高了19.72%，而参与度实时指标略微下降了-0.22%。其次，通过使用MMoE，与Shared-Bottom模型相比，我们提高了两个指标。在这个推荐系统中，参与度指标的原始值远大于满意度指标，因此在提高满意度指标的同时，不希望参与度指标有任何损失，甚至希望有所提高。
>
> 这段文字主要描述了在线实验的过程和结果，包括实验的设计、实验模型的选择以及实验结果的分析。通过对比不同模型在参与度和满意度两个指标上的表现，来评估模型的性能。

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/multitask/MMOE/Table4.png)

## 7 CONCLUSION

We propose a novel multi-task learning approach, Multi-gate MoE (MMoE), that explicitly learns to model task relationship from data. We show by control experiments on synthetic data that the proposed approach can better handle the scenario where tasks are less related. We also show that the MMoE is easier to train compared to baseline methods. With experiments on benchmark dataset and a real large-scale recommendation system, we demonstrate the success of the proposed method over several state-of-the-art baseline multi-task learning models.

Besides the benets above, another major design consideration in real machine learning production systems is the computational eciency. This is also one of the most important reasons that the Shared-Bottom multi-task model is widely used. The shared part of the model saves a lot of computation at serving time [18, 29]. All of the three state-of-the-art baseline models (see section 6.1) learn the task relationship at the loss of this computational benet. The MMoE model, however, largely preserves the computational advantage since the gating networks are usually light-weight and the expert networks are shared across all the tasks. Moreover, this model has the potential to achieve even better computational e- ciency by making the gating network as a sparse top-k gate [31]. We hope this work inspire other researchers to further investigate multi-task modeling using these approaches.

> 我们提出了一种新颖的多任务学习方法，即多门混合专家模型（MMoE），该方法能够明确地从数据中学习并建模任务之间的关系。通过合成数据上的对照实验，我们展示了所提出的方法能够更好地应对任务之间相关性较低的情况。此外，我们还证明了与基线方法相比，MMoE更容易进行训练。通过在基准数据集和真实的大规模推荐系统上进行实验，我们验证了该方法相对于几种最先进的基线多任务学习模型的优势和成功。
>
> 除了上述优点之外，实际机器学习生产系统中的另一个主要设计考量是计算效率。这也是共享底层（Shared-Bottom）多任务模型被广泛使用的重要原因之一。模型的共享部分在服务时节省了大量的计算资源。所有三种最先进的基线模型（见6.1节）在学习任务关系时，都牺牲了这种计算优势。然而，MMoE模型在很大程度上保留了计算优势，因为门控网络通常较轻量级，并且专家网络在所有任务之间是共享的。此外，通过将门控网络设置为稀疏的top-k门，该模型有可能实现更好的计算效率。我们希望这项工作能激发其他研究人员使用这些方法进一步研究多任务建模。