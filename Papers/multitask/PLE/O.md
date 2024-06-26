## Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

## ABSTRACT

Multi-task learning (MTL) has been successfully applied to many recommendation applications. However, MTL models often suffer from performance degeneration with negative transfer due to the complex and competing task correlation in real-world recommender systems. Moreover, through extensive experiments across SOTA MTL models, we have observed an interesting seesaw phenomenon that performance of one task is often improved by hurting the performance of some other tasks. To address these issues, we propose a Progressive Layered Extraction (PLE) model with a novel sharing structure design. PLE separates shared components and task-specific components explicitly and adopts a progressive routing mechanism to extract and separate deeper semantic knowledge gradually, improving efficiency of joint representation learning and information routing across tasks in a general setup. We apply PLE to both complicatedly correlated and normally correlated tasks, ranging from two-task cases to multi-task cases on a real-world Tencent video recommendation dataset with 1 billion samples, and results show that PLE outperforms state-of-the-art MTL models significantly under different task correlations and task-group size. Furthermore, online evaluation of PLE on a large-scale content recommendation platform at Tencent manifests 2.23% increase in view-count and 1.84% increase in watch time compared to SOTA MTL models, which is a significant improvement and demonstrates the effectiveness of PLE. Finally, extensive offline experiments on public benchmark datasets demonstrate that PLE can be applied to a variety of scenarios besides recommendations to eliminate the seesaw phenomenon. PLE now has been deployed to the online video recommender system in Tencent successfully.

> 多任务学习（MTL）已成功应用于许多推荐应用中。然而，由于现实世界推荐系统中的任务相关性复杂且存在竞争关系，MTL模型经常因negative transfer而性能退化。此外，通过对最先进的MTL模型进行大量实验，我们观察到了一个有趣的跷跷板现象，即一个任务性能的提升往往是通过损害其他某些任务的性能来实现的。为了解决这些问题，我们提出了一种具有新型共享结构设计的渐进式分层提取（PLE）模型。PLE明确地将共享组件和任务特定组件分开，并采用渐进式路由机制逐步提取和分离更深层次的语义知识，从而在一般设置中提高联合表示学习和跨任务信息路由的效率。我们将PLE应用于复杂相关和正常相关的任务，从两任务案例到多任务案例，在一个包含10亿样本的腾讯真实视频推荐数据集上进行测试。结果表明，在不同任务相关性和任务组大小的情况下，PLE的性能明显优于最先进的MTL模型。此外，与最先进的MTL模型相比，腾讯大型内容推荐平台上的PLE在线评估显示，观看次数增加了2.23%，观看时间增加了1.84%，这是一个显著的改进，证明了PLE的有效性。最后，在公开基准数据集上进行的大量离线实验表明，PLE除了推荐场景外，还可以应用于多种场景，以消除跷跷板现象。目前，PLE已成功部署到腾讯的在线视频推荐系统中。

## 1 INTRODUCTION

Personalized recommendation has played a crucial role in online applications. Recommender systems (RS) need to incorporate various user feedbacks to model user interests and maximize user engagement and satisfaction. However, user satisfaction is normally hard to tackle directly by a learning algorithm due to the high dimensionality of the problem. Meanwhile, user satisfaction and engagement have many major factors that can be learned directly, e.g. the likelihood of clicking, finishing, sharing, favoriting, and commenting etc. Therefore, there has been an increasing trend to apply Multi-Task Learning (MTL) in RS to model the multiple aspects of user satisfaction or engagement simultaneously. And in fact, it has been the mainstream approach in major industry applications[11, 13, 14, 25].

> 个性化推荐在线应用中扮演着至关重要的角色。推荐系统（RS）需要融合各种用户反馈来建模用户兴趣，并最大化用户参与度和满意度。然而，由于问题的高维度，用户满意度通常很难直接通过学习算法来解决。同时，用户满意度和参与度有许多可以直接学习的主要因素，例如点击、完成、分享、收藏和评论等的可能性。
>
> 因此，在推荐系统中应用多任务学习（MTL）来同时建模用户满意度或参与度的多个方面，已成为一种增长的趋势。事实上，这已经成为主要行业应用中的主流方法。

MTL learns multiple tasks simultaneously in one single model and is proven to improve learning efficiency through information sharing between tasks [2]. However, tasks in real-world recommender systems are often loosely correlated or even conflicted, which may lead to performance deterioration known as negative transfer [21]. Through extensive experiments in a real-world largescale video recommender system and public benchmark datasets, we find that existing MTL models often improve some tasks at the sacrifice of the performance of others, when task correlation is complex and sometimes sample dependent, i.e., multiple tasks could not be improved simultaneously compared to the corresponding single-task model, which is called seesaw phenomenon in this paper.

> 多任务学习（MTL）可以在一个模型中同时学习多个任务，并通过任务间的信息共享提高学习效率。然而，在现实世界的推荐系统中，任务之间往往只是松散相关，甚至可能存在冲突，这可能导致性能下降，这种现象被称为负迁移。通过在一个真实的大规模视频推荐系统和公共基准数据集上进行大量实验，我们发现，当任务相关性复杂且有时依赖于样本时，现有的多任务学习模型在提高某些任务性能的同时，往往会牺牲其他任务的性能。也就是说，与相应的单任务模型相比，多个任务不能同时得到改进，本文称这种现象为“跷跷板现象”。

Prior works put more efforts to address the negative transfer but neglect the seesaw phenomenon, e.g., cross-stitch network [16] and sluice network [18] propose to learn static linear combinations to fuse representations of different tasks, which could not capture the sample dependence. MMOE [13] applies gating networks to combine bottom experts based on the input to handle task differences but neglects the differentiation and interaction between experts, which is proved to suffer from the seesaw phenomenon in our industrial practice. Hence, it is critical to design a more powerful and efficient model to handle complicated correlations and eliminate the challenging seesaw phenomenon.

> 先前的工作更多地致力于解决负迁移问题，但忽视了跷跷板现象。例如，cross-stitch网络和sluice网络提出了学习静态线性组合来融合不同任务的表示，但这不能捕获样本依赖性。MMOE应用门控网络来根据输入组合底层专家以处理任务差异，但忽视了专家之间的区别和交互，这在我们的工业实践中被证明会受到跷跷板现象的影响。因此，设计一个更强大、更高效的模型来处理复杂的相关性并消除具有挑战性的跷跷板现象是至关重要的。

To achieve this goal, we propose a novel MTL model called Progressive Layered Extraction (PLE), which better exploits prior knowledge in the design of shared network to capture complicated task correlations. Compared with roughly shared parameters in MMOE, PLE explicitly separates shared and task-specific experts to alleviate harmful parameter interference between common and task-specific knowledge. Furthermore, PLE introduces multi-level experts and gating networks, and applies progressive separation routing to extract deeper knowledge from lower-layer experts and separate task-specific parameters in higher levels gradually.

> 为了实现这一目标，我们提出了一种新的多任务学习模型，称为渐进式分层提取（PLE），该模型在设计共享网络时更好地利用了先验知识，以捕获复杂的任务相关性。与MMOE中的粗略共享参数相比，PLE明确地将共享专家和特定任务专家分开，以减轻公共知识和特定任务知识之间的有害参数干扰。此外，PLE引入了多级专家和门控网络，并应用渐进式分离路由从低层专家中提取更深层次的知识，并在更高层次上逐步分离特定任务的参数。

To evaluate the performance of PLE, we conduct extensive experiments on real-world industrial recommendation dataset and major available public datasets including census-income [5], synthetic data [13] and Ali-CCP 1 . Experiment results demonstrate that PLE outperforms state-of-the-art MTL models across all datasets, showing consistent improvements on not only task groups with challenging complex correlations, but also task groups with normal correlations in different scenarios. Besides, significant improvement of online metrics on a large-scale video recommender system in Tencent demonstrates the advantage of PLE in real-world recommendation applications.

> 为了评估PLE的性能，我们在真实的工业推荐数据集和主要的可用公共数据集上进行了广泛的实验，包括人口普查收入数据、合成数据和Ali-CCP。实验结果表明，PLE在所有数据集上的表现都优于最先进的MTL模型，不仅在具有挑战性复杂相关性的任务组上表现出一致的改进，而且在不同场景中具有正常相关性的任务组上也表现出一致的改进。此外，腾讯大规模视频推荐系统的在线指标显著提升，证明了PLE在现实推荐应用中的优势。

The main contributions of this paper are summarized as follows:

- Through extensive experiments in the large-scale video recommender system at Tencent and public benchmark datasets, an interesting seesaw phenomenon has been observed that SOTA MTL models often improve some tasks at the sacrifice of the performance of others and do not outperform the corresponding single-task model due to the complicatedly inherent correlations.
- A PLE model with novel shared learning structure is proposed to improve shared learning efficiency then address the seesaw phenomenon and negative transfer further, from the perspective of joint representation learning and information routing. Besides recommendation applications, PLE is flexible to be applied to a variety of scenarios.
- Extensive offline experiments are conducted to evaluate the effectiveness of PLE on industrial and public benchmark datasets. Online A/B test results in one of the world’s largest content recommendation platforms at Tencent also demonstrate the significant improvement of PLE over SOTA MTL models in real-world applications, with 2.23% increase in view-count and 1.84% increase in watch time, which generates significant business revenue. PLE has been successfully deployed to the recommender system now and can be potentially applied to many other recommendation applications.

> 本文的主要贡献概括如下：
>
> - 通过在腾讯大规模视频推荐系统和公共基准数据集上进行大量实验，我们观察到了一个有趣的跷跷板现象，即由于任务间复杂固有的相关性，最先进的MTL模型在提高某些任务性能的同时，往往会牺牲其他任务的性能，并且表现并不优于相应的单任务模型。
> - 从联合表示学习和信息路由的角度出发，提出了一种具有新颖共享学习结构的PLE模型，以提高共享学习效率，从而进一步解决跷跷板现象和负迁移问题。除了推荐应用外，PLE还可以灵活应用于多种场景。
> - 我们进行了广泛的离线实验，以评估PLE在工业和公共基准数据集上的有效性。在腾讯世界上最大的内容推荐平台之一进行的在线A/B测试结果也表明，在实际应用中，PLE相较于最先进的MTL模型有显著提升，观看次数增加了2.23%，观看时间增加了1.84%，从而带来了显著的业务收入。PLE现已成功部署到推荐系统中，并有可能应用于许多其他推荐应用中。

## 2 RELATED WORK

Efficient multi-task learning models and application of MTL models in recommender systems are two research areas related to our work. In this section, we briefly discuss related works in these two areas.

> 高效的多任务学习模型以及多任务学习模型在推荐系统中的应用是与我们的工作相关的两个研究领域。在本节中，我们将简要讨论这两个领域的相关工作。

#### 2.1 Multi-Task Learning Models

Hard parameter sharing [2] shown in Fig. 1a) is the most basic and commonly used MTL structure but may suffer from negative transfer due to task conflicts as parameters are straightforwardly shared between tasks. To deal with task conflicts, cross-stitch network [16] in Fig. 1f) and sluice network [18] in Fig. 1g) both propose to learn weights of linear combinations to fuse representations from different tasks selectively. However, representations are combined with the same static weights for all samples in these models and the seesaw phenomenon is not addressed. In this work, the proposed PLE (Progressive Layered Extraction) model applies progressive routing mechanism with gate structures to fuse knowledge based on the input, which achieves adaptive combinations for different inputs.

> 如图1a）所示，硬参数共享是最基本且最常用的多任务学习结构，但由于任务之间的参数直接共享，可能会因任务冲突而导致负迁移。为了处理任务冲突，图1f）中的cross-stitch网络和图1g）中的sluice网络都提出了学习线性组合的权重，以有选择地融合来自不同任务的表示。然而，在这些模型中，所有样本的表示都使用相同的静态权重进行组合，并且没有解决跷跷板现象。在这项工作中，提出的PLE（渐进式分层提取）模型使用带门结构的渐进路由机制来基于输入融合知识，从而实现了针对不同输入的自适应组合。

There have been some studies applying the gate structure and attention network for information fusion. MOE [8] first proposes to share some experts at the bottom and combine experts through a gating network. MMOE [13] extends MOE to utilize different gates for each task to obtain different fusing weights in MTL. Similarly, MRAN [24] applies multi-head self-attention to learn different representation subspaces at different feature sets. The expert and attention module are shared among all tasks and there is no taskspecific concept in MOE, MMOE (shown in Fig. 1) and MRAN. In contrast, our proposed CGC (Customized Gate Control) and PLE model separate task-common and task-specific parameters explicitly to avoid parameter conflicts resulted from complex task correlations. Even though there exists theoretical possibility for MMOE to converge to our network design, the prior knowledge on network design is important and MMOE can hardly discover the convergence path in practice. Liu et al. [10] apply task-specific attention networks to fuse shared features selectively but different tasks still share the same representation before fusion in attention network. None of the previous works has explicitly addressed the issues of joint optimization of representation learning and routing, especially in an inseparable joint fashion, while this work makes the first effort to propose a novel progressive separation fashion on the general framework of joint learning and routing.

> 已经有一些研究将门结构和注意力网络应用于信息融合。MOE首次提出了在底部共享一些专家，并通过门控网络组合专家。MMOE将MOE扩展到利用不同的门为每个任务在MTL中获得不同的融合权重。类似地，MRAN应用多头自注意力在不同的特征集上学习不同的表示子空间。所有任务都共享专家和注意力模块，并且在MOE、MMOE（如图1所示）和MRAN中没有特定于任务的概念。
>
> 相比之下，我们提出的CGC（定制门控制）和PLE模型明确地将任务公共和任务特定参数分开，以避免由复杂的任务相关性导致的参数冲突。尽管MMOE在理论上存在收敛到我们网络设计的可能性，但网络设计的先验知识很重要，MMOE在实践中很难发现收敛路径。Liu等人应用特定于任务的注意力网络来有选择地融合共享特征，但在注意力网络中融合之前，不同的任务仍然共享相同的表示。以前的工作都没有明确解决表示学习和路由的联合优化问题，特别是在不可分割的联合方式中，而这项工作首次努力在联合学习和路由的一般框架上提出了一种新颖的渐进分离方式。

There have also been some works utilizing AutoML approaches to find a good network structure. SNR framework [12] controls connections between sub-networks by binary random variables and applies NAS [26] to search for the optimal structure. Similarly, Gumbel-matrix routing framework [15] learns routing of MTL models formulated as a binary matrix with Gumbel-Softmax trick. Modeling routing process as MDP, Rosenbaum et al. [17] applies MARL [19] to train the routing network. The network structures in these works are designed with certain simplified assumptions and are not general enough. The routing network in [17] selects no more than one function block for each task in each depth, which reduces the expressivity of the model. Gumbel-matrix routing network [15] imposes the constraint on the representation learning as each task’s input needs to merge to one representation at each layer. Moreover,the fusing weights in these frameworks are not adjustable for different inputs, and the expensive searching cost is another challenge for these approaches to find the optimal structure.

> 还有一些工作利用AutoML方法来寻找良好的网络结构。SNR框架通过二进制随机变量控制子网络之间的连接，并应用NAS来搜索最优结构。类似地，Gumbel-matrix路由框架将MTL模型的路由建模为一个二进制矩阵，并使用Gumbel-Softmax技巧进行学习。Rosenbaum等人将路由过程建模为MDP，并应用MARL来训练路由网络。这些作品中的网络结构是基于某些简化的假设设计的，并不够通用。[17]中的路由网络在每个深度为每个任务选择的功能块不超过一个，这降低了模型的表达力。Gumbel-matrix路由网络对每个任务的输入在每一层都需要合并为一个表示，从而对表示学习施加了约束。此外，这些框架中的融合权重对于不同的输入是不可调整的，而昂贵的搜索成本是这些方法找到最优结构的另一个挑战。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure1.png)

#### 2.2 Multi-Task Learning in Recommender Systems

To better exploit various user behaviors, multi-task learning has been widely applied to recommender systems and achieved substantial improvement. Some studies integrate traditional recommendation algorithms such as collaborative filtering and matrix factorization with MTL. Lu et al. [11] and Wang et al. [23] impose regularization on latent representations learned for the recommendation task and explanation task to optimize them jointly. Wang et al. [22] combine collaborative filtering with MTL to learn user-item similarity more efficiently. Compared to PLE in this paper, these factorization based models exhibit lower expressivity and could not fully exploit commonalities between tasks.

>为了更好地利用各种用户行为，多任务学习已广泛应用于推荐系统，并取得了实质性改进。一些研究将协同过滤和矩阵分解等传统推荐算法与多任务学习（MTL）相结合。Lu等人[11]和Wang等人[23]对推荐任务和解释任务学习到的潜在表示施加正则化，以联合优化它们。Wang等人[22]将协同过滤与多任务学习相结合，以更有效地学习用户-项目的相似性。与本文中的PLE相比，这些基于分解的模型表现出较低的表达能力，并且无法充分利用任务之间的共性。

As the most basic MTL structure, hard parameter sharing has been applied to many DNN based recommender systems. The ESSM [14] introduces two auxiliary tasks of CTR (Click-Through Rate) and CTCVR and shares embedding parameters between CTR and CVR (Conversion Rate) to improve the performance of CVR prediction. Hadash et al. [7] propose a multi-task framework to learn parameters of the ranking task and rating task simultaneously. The task of text recommendation in [1] is improved through sharing representations at the bottom. However, hard parameter sharing often suffers from negative transfer and seesaw phenomenon under loose or complex task correlations. In contrast, our proposed model introduces a novel sharing mechanism to achieve more efficient information sharing in general.

> 作为最基本的MTL（多任务学习）结构，硬参数共享已应用于许多基于深度神经网络的推荐系统。ESSM模型引入了点击率（CTR）和点击转化率（CTCVR）这两个辅助任务，并在CTR和转化率（CVR）之间共享嵌入参数，以提高CVR预测的性能。Hadash等人提出了一个多任务框架，以同时学习任务排序和评分的参数。通过底部共享表示，可以改善文本推荐任务的效果。然而，在任务相关性松散或复杂的情况下，硬参数共享经常会受到负迁移和跷跷板现象的影响。相比之下，我们提出的模型引入了一种新颖的共享机制，以实现更高效的信息共享。

Besides hard parameter sharing, there have been some recommender systems applying MTL models with more efficient shared learning mechanism. To better exploit correlations between tasks, Chen et al. [3] utilize hierarchical multi-pointer co-attention [20] to improve the performance of the recommendation task and explanation task. However, tower networks of each task share the same representation in the model, which may still suffer from task conflicts. Applying MMOE [13] to combine shared experts through different gating networks for each task, the YouTube video recommender system in [25] can better capture task differences and optimize multiple objectives efficiently. Compared with MMOE which treats all experts equally without differentiation, PLE in this paper explicitly separates task-common and task-specific experts and adopts a novel progressive separation routing to achieve significant improvement over MMOE in real-world video recommender systems.

> 除了硬参数共享外，还有一些推荐系统应用了具有更高效共享学习机制的多任务学习（MTL）模型。为了更好地利用任务之间的相关性，Chen等人利用hierarchical multi-pointer co-attention来提高推荐任务和解释任务的性能。然而，该模型中每个任务的塔式网络共享相同的表示，这可能仍然会受到任务冲突的影响。通过应用MMOE（Multi-gate Mixture-of-Experts）结合不同门控网络为每个任务组合共享专家，YouTube视频推荐系统可以更好地捕捉任务差异并有效地优化多个目标。与MMOE相比，后者对所有专家一视同仁，没有区别对待，本文中的PLE（Progressive Layered Extraction）模型明确地区分了任务通用和任务特定专家，并采用了一种新颖的逐步分离routing，在实际视频推荐系统中实现了对MMOE的显著改进。

## 3 SEESAW PHENOMENON IN MTL FOR RECOMMENDATION

Negative transfer is a common phenomenon in MTL especially for loosely correlated tasks [21]. For complex task correlation and especially sample dependent correlation patterns, we also observe the seesaw phenomenon where improving shared learning efficiency and achieving significant improvement over the corresponding single-task model across all tasks is difficult for current MTL models. In this section, we introduce and investigate the seesaw phenomenon thoroughly based on a large-scale video recommender system in Tencent.

> 负迁移是多任务学习（MTL）中的常见现象，特别是在任务相关性不高的情况下。对于复杂的任务相关性，尤其是样本依赖的相关性模式，我们还观察到跷跷板现象，即在当前的多任务学习模型中，提高共享学习效率并在所有任务上相对于对应的单任务模型实现显著提升是困难的。在本节中，我们将基于腾讯的大规模视频推荐系统对跷跷板现象进行深入介绍和研究。

## 3.1 An MTL Ranking System for Video Recommendation

In this subsection, we briefly introduce the MTL ranking system serving Tencent News, which is one of the world’s largest content platforms and recommends news and videos to users based on the diverse user feedbacks. As shown in Fig. 2, there are multiple objectives to model different user behaviors such as click, share, and comment in the MTL ranking system. In the offline training process, we train the MTL ranking model based on user actions extracted from user logs. After each online request, the ranking model outputs predictions for each task, then the weighted-multiplication based ranking module combines these predicted scores to a final score through a combination function shown in Equation 1, and recommends top-ranked videos to the user finally.
$$
\begin{array}{r}
\text { score }=p_{V T R}{ }^{w_{V T R}} \times p_{V C R}{ }^{w_{V C R}} \times p_{S H R}{ }^{w_{S H R}} \times \cdots \times \\
p_{C M R}{ }^{w_{C M R}} \times f(\text { video\_len }),
\end{array}
$$


> 在本小节中，我们简要介绍为腾讯新闻服务的多任务学习（MTL）排序系统。腾讯新闻是世界上最大的内容平台之一，它根据用户的多种反馈为用户推荐新闻和视频。如图2所示，在MTL排序系统中存在多个目标来模拟不同的用户行为，如点击、分享和评论。在离线训练过程中，我们基于从用户日志中提取的用户行为来训练MTL排序模型。每次在线请求后，排序模型会针对每个任务输出预测结果，然后基于加权乘法的排序模块使用公式1中的组合函数将这些预测分数组合成一个最终分数，并最终向用户推荐排名最高的视频。
> $$
> \begin{array}{r}
> \text { score }=p_{V T R}{ }^{w_{V T R}} \times p_{V C R}{ }^{w_{V C R}} \times p_{S H R}{ }^{w_{S H R}} \times \cdots \times \\
> p_{C M R}{ }^{w_{C M R}} \times f(\text { video\_len }),
> \end{array}
> $$

where each $w$ determines the relative importance of each predicted score, $f (video\_len)$ is a non-linear transform function such as sigmoid or log function in video duration.$w_{V T R}$,$w_{V C R}$, $w_{S H R}$,$w_{CMR}$ are hyper-paramters optimized through online experimental search to maximize online metrics.

> 在这个公式中，每个$w$表示各个预测分数的相对重要性，$f(video\_len)$是一个关于视频时长的非线性转换函数，例如sigmoid函数或对数函数。$w_{VTR}$、$w_{VCR}$、$w_{SHR}$、$w_{CMR}$是通过在线实验搜索来优化的超参数，目的是最大化在线评估指标。
>

Out of all tasks, VCR (View Completion Ratio) and VTR (ViewThrough Rate) are two important objectives modeling key online metrics of view-count and watch time respectively. Specifically,VCR prediction is a regression task trained with MSE loss to predict the completion ratio of each view. VTR prediction is a binary classification task trained with cross-entropy loss to predict the probability of a valid view, which is defined as a play action that exceeds a certain threshold of watch time. The correlation pattern between VCR and VTR is complex. First, the label of VTR is a coupled factor of play action and VCR, as only a play action with watch time exceeding the threshold will be treated as a valid view. Second, the distribution of play action is further complicated as samples from auto-play scenarios in WIFI exhibit higher average probability of play, while other samples from explicit click scenarios without auto-play exhibit lower probability of play. Due to the complex and strong sample dependent correlation pattern, a seesaw phenomenon is observed when modeling VCR and VTR jointly.

> 在所有任务中，观看完成率（View Completion Ratio，VCR）和观看率（ViewThrough Rate，VTR）是两个重要的目标，分别对应在线指标观看次数和观看时间的关键建模。具体来说，VCR预测是一个回归任务，使用均方误差（MSE）损失进行训练，以预测每次观看的完成率。VTR预测是一个二元分类任务，使用交叉熵损失进行训练，以预测有效观看的概率，有效观看定义为观看时间超过特定阈值的播放行为。
>
> VCR和VTR之间的相关性模式是复杂的。首先，VTR的标签是播放行为和VCR的耦合因素，因为只有观看时间超过阈值的播放行为才会被视为有效观看。其次，播放行为的分布更加复杂，因为来自WIFI环境下自动播放场景的样本展现出更高的平均播放概率，而来自没有自动播放功能的显式点击场景的样本则展现出较低的播放概率。
>
> 由于这种复杂且强烈的样本依赖相关性模式，在联合建模VCR和VTR时会出现跷跷板现象。这意味着当一个任务的性能提升时，另一个任务的性能可能会下降，因为它们之间存在复杂的相互依赖关系，而且不同样本群体之间的这种关系可能有所不同。因此，在优化多任务学习模型时需要仔细平衡这两个目标，以确保整体性能的提升。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure2.png)

#### 3.2 Seesaw Phenomenon in MTL

To better understand the seesaw phenomenon, we perform experimental analysis with the single-task model and SOTA MTL models on the complicatedly correlated task-group of VCR and VTR in our ranking system. Besides hard parameter sharing, cross-stitch [16], sluice network [18] and MMOE [13], we also evaluate two innovatively proposed structures called asymmetric sharing and customized sharing:

- Asymmetric Sharing is a novel sharing mechanism to capture asymmetric relations between tasks. According to Fig. 1b), bottom layers are shared asymmetrically between tasks, and representation of which task to be shared depends on relations between tasks. Common fusion operations such as concatenation, sumpooling, and average-pooling can be applied to combine outputs of bottom layers of different tasks.

- Customized Sharing shown in Fig. 1c) separates shared and task-specific parameters explicitly to avoid inherent conflicts and negative transfer. Compared with the single-task model, customized sharing adds a shared bottom layer to extract sharing information and feeds the concatenation of the shared bottom layer and task-specific layer to the tower layer of the corresponding task.

> 为了更好地理解跷跷板现象，我们在排名系统中对 VCR 和 VTR 这两个复杂相关任务组进行了实验分析，使用了单任务模型和最先进的MTL模型。除了硬参数共享、Cross-stitch网络、Sluice网络和MMOE模型外，我们还评估了两种新提出的结构，即不对称共享和定制共享。
>
> - 不对称共享是一种新型的共享机制，用于捕获任务之间的不对称关系。如图1b所示，底层在任务之间是不对称共享的，要共享哪个任务的表示取决于任务之间的关系。常见的融合操作，如连接、和池化和平均池化，可以用于组合不同任务底层的输出。
> - 如图1c所示，定制共享明确地区分了共享参数和任务特定参数，以避免固有的冲突和负迁移。与单任务模型相比，定制共享添加了一个共享的底层来提取共享信息，并将共享的底层和任务特定层的连接提供给相应任务的塔层。

Fig.3 illustrates experiment results, where bubbles closer to upper-right indicate better performance with higher AUC and lower MSE. It is worth noting that 0.1% increase of AUC or MSE contributes significant improvement to online metrics in our system, which is also mentioned in [4, 6, 14]. One can see that hard parameter sharing and cross-stitch network suffer from significant negative transfer and perform worst in VTR. Through innovative sharing mechanism to capture asymmetric relations, asymmetric sharing achieves significant improvement in VTR but exhibits significant degeneration in VCR, similar to sluice network. Owing to explicit separation of shared layers and task-specific layers, customized sharing improves VCR over the single-task model while still slightly suffers in VTR. MMOE improves over the single-task model on both tasks but the improvement of VCR is only +0.0001 on the borderline. Although these models exhibit different learning efficiency with these two challenging tasks, we clearly observe the seesaw phenomenon that the improvement of one task often leads to performance degeneration of the other task, as no one baseline MTL model lies in 2nd quadrant completely. Experiments with SOTA models on public benchmark datasets also exhibit clear seesaw phenomenon. Details would be provided in Section 5.2.

> 图3展示了实验结果，其中靠近右上角的气泡表示性能更好，AUC更高，MSE更低。值得注意的是，在我们的系统中，AUC或MSE提高0.1%都会对在线指标产生显著的改善，这在[4, 6, 14]中也有提及。可以看出，硬参数共享和cross-stitch网络存在明显的负迁移，在VTR上表现最差。通过创新的共享机制来捕获不对称关系，不对称共享在VTR上实现了显著的改善，但在VCR上表现出明显的退化，类似于sluice网络。由于共享层和任务特定层的明确分离，定制共享在单任务模型的基础上提高了VCR，但在VTR上仍然略有不足。MMOE在两个任务上都优于单任务模型，但VCR的提升仅在边界线上为+0.0001。尽管这些模型在这两个具有挑战性的任务上表现出不同的学习效率，但我们清楚地观察到了跷跷板现象，即一个任务的改进往往会导致另一个任务性能的退化，因为没有一个基线MTL模型完全位于第二象限。在公共基准数据集上使用SOTA模型的实验也表现出了明显的跷跷板现象。详细信息将在5.2节中提供。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure3.png)

As aforementioned, the correlation pattern between VCR and VTR is complicated and sample dependent. Specifically, there are some partially ordered relations between VCR and VTR, and different samples exhibit different correlations. Thus, cross-stitch and sluice network that combine shared representations with same static weights for all samples could not capture the sample dependence and suffer from the seesaw phenomenon. Applying gates to obtain fusing weights based on the input, MMOE handles task difference and sample difference to some extent, which outperforms other baseline MTL models. Nevertheless, experts are shared among all tasks in MMOE without differentiation, which could not capture the complicated task correlations and may bring harmful noise to some tasks. Moreover, MMOE neglects the interactions between different experts, which limits the performance of joint optimization further. In addition to VCR and VTR, there are many complicatedly correlated tasks in industrial recommendation applications as human behaviors are often subtle and complex, e.g., CTR prediction and CVR prediction in online advertising and e-commerce platform [14]. Therefore, a powerful network that considers differentiation and interactions between experts is critical to eliminate the challenging seesaw phenomenon resulted from complex task correlation.

> 如前所述，VCR和VTR之间的相关性模式是复杂的，并且与样本有关。具体来说，VCR和VTR之间存在一些偏序关系，不同的样本表现出不同的相关性。因此，对所有样本使用相同静态权重的共享表示的cross-stitch和sluice网络无法捕获样本依赖性，并会受到跷跷板现象的影响。通过应用门控机制基于输入获得融合权重，MMOE在一定程度上处理了任务差异和样本差异，其性能优于其他基线MTL模型。然而，MMOE中的所有任务都共享专家，没有差异化，这无法捕获复杂的任务相关性，并可能给某些任务带来有害的噪声。此外，MMOE忽视了不同专家之间的交互，这进一步限制了联合优化的性能。除了VCR和VTR之外，工业推荐应用中还有许多复杂相关的任务，因为人类行为往往是微妙和复杂的，例如在线广告和电子商务平台上的点击率预测和转化率预测。因此，一个考虑专家之间差异化和交互的强大网络对于消除由复杂任务相关性导致的挑战性跷跷板现象至关重要。

In this paper, we propose a Progressive Layered Extraction (PLE) model to address the seesaw phenomenon and negative transfer. The key idea of PLE is as follows. First, it explicitly separates shared and task-specific experts to avoid harmful parameter interference. Second, multi-level experts and gating networks are introduced to fuse more abstract representations. Finally, it adopts a novel progressive separation routing to model interactions between experts and achieve more efficient knowledge transferring between complicatedly correlated tasks. As shown in Fig. 3, PLE achieves significant improvement over MMOE in both tasks. Details of structure design and experiments would be described in Section 4 and Section 5 respectively.

> 在本文中，我们提出了一种渐进式分层提取（PLE）模型，以解决跷跷板现象和负迁移问题。
>
> PLE的关键思想如下。首先，它明确地将共享专家和特定于任务的专家分开，以避免有害的参数干扰。其次，引入了多级专家和门控网络来融合更抽象的表示。最后，它采用了一种新颖的渐进式分离路由来建模专家之间的交互，并在复杂相关任务之间实现更有效的知识传递。如图3所示，PLE在两个任务中都显著优于MMOE。结构设计和实验的详细信息将分别在第4节和第5节中描述。

## 4 PROGRESSIVE LAYERED EXTRACTION

To address the seesaw phenomenon and negative transfer, we propose a Progressive Layered Extraction (PLE) model with a novel sharing structure design in this section. First, a Customized Gate Control (CGC) model that explicitly separates shared and taskspecific experts is proposed. Second, CGC is extended to a generalized PLE model with multi-level gating networks and progressive separation routing for more efficient information sharing and joint learning. Finally, the loss function is optimized to better handle the practical challenges of joint training for MTL models.

> 为了应对跷跷板现象和负迁移问题，我们在本节中提出了一种具有新颖共享结构设计的渐进式分层提取（PLE）模型。首先，我们提出了一个定制门控（CGC）模型，该模型明确地将共享专家和特定于任务的专家分开。其次，我们将CGC扩展到广义的PLE模型，该模型具有多级门控网络和渐进式分离路由，以实现更高效的信息共享和联合学习。最后，我们优化了损失函数，以更好地应对多任务学习（MTL）模型联合训练的实际挑战。

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure4.png)

#### 4.1 Customized Gate Control

Motivated by customized sharing which achieves similar performance with the single-task model through explicitly separating shared and task-specific layers, we first introduce a Customized Gate Control (CGC) model. As shown in Fig. 4, there are some expert modules at the bottom and some task-specific tower networks at the top. Each expert module is composed of multiple sub-networks called experts and the number of experts in each module is a hyperparameter to tune. Similarly, a tower network is also a multi-layer network with width and depth as hyper-parameters. Specifically, the shared experts in CGC are responsible for learning shared patterns, while patterns for specific tasks are extracted by task-specific experts. Each tower network absorbs knowledge from both shared experts and its own task-specific experts, which means that the parameters of shared experts are affected by all tasks while parameters of task-specific experts are only affected by the corresponding specific task.



> 受定制共享机制的启发（该机制通过明确分离共享层和特定于任务的层来实现与单任务模型相似的性能），我们首先引入了一个定制门控（CGC）模型。如图4所示，底部有一些专家模块，顶部有一些特定于任务的塔式网络。每个专家模块都由多个称为专家的子网络组成，每个模块中的专家数量是一个可调的超参数。类似地，塔式网络也是一个多层网络，其宽度和深度作为超参数。具体来说，CGC中的共享专家负责学习共享模式，而特定任务的模式则由特定于任务的专家提取。每个塔式网络都从共享专家和其自身的特定于任务的专家那里吸收知识，这意味着共享专家的参数受所有任务的影响，而特定于任务的专家的参数仅受相应特定任务的影响。

In CGC, shared experts and task-specific experts are combined through a gating network for selective fusion. As depicted in Fig. 4, the structure of the gating network is based on a single-layer feedforward network with SoftMax as the activation function, input as the selector to calculate the weighted sum of the selected vectors, i.e., the outputs of experts. More precisely, the output of task $k
$’s gating network is formulated as:
$$
g^k(x)=w^k(x) S^k(x)
$$
where $x$ is the input representation, and $w^k (x)$ is a weighting function to calculate the weight vector of task $k$ through linear transformation and a SoftMax layer:
$$
w^k(x)=\operatorname{Softmax}\left(W_g^k x\right)
$$
where $W_q^k \in R^{\left(m_k+m_s\right) \times d}$ is a parameter matrix, $m_s$ and $m_k$ are the number of shared experts and task $k$’s specific experts respectively, $d$ is the dimension of input representation. $S^k (x)$ is a selected matrix composed of all selected vectors including shared experts and task $k$’s specific experts:
$$
S^k(x)=\left[E_{(k, 1)}^T, E_{(k, 2)}^T, \ldots, E_{\left(k, m_k\right)}^T, E_{(s, 1)}^T, E_{(s, 2)}^T, \ldots, E_{\left(s, m_s\right)}^T\right]^T
$$
Finally, the prediction of task $k$ is:
$$
y^k(x)=t^k\left(g^k(x)\right)
$$
where $t^k$ denotes the tower network of task $k$.

Compared with MMOE, CGC removes connections between a task’s tower network and task-specific experts of other tasks, enabling different types of experts to concentrate on learning different knowledge efficiently without interference. Combined with the benefit of gating networks to fuse representations dynamically based on the input, CGC achieves more flexible balance between tasks and better deals with task conflicts and sample-dependent correlations.

> 在CGC中，共享专家和特定于任务的专家通过一个门控网络进行选择性融合。如图4所示，门控网络的结构基于单层前馈网络，使用 SoftMax 作为激活函数，输入作为选择器来计算选定向量的加权和，即专家的输出。更准确地说，任务 $k$ 的门控网络的输出公式为：
>
> $$
> g^k(x)=w^k(x) S^k(x)
> $$
> 其中 $x$ 是输入表示，而 $w^k (x)$ 是一个加权函数，通过线性变换和 SoftMax 层来计算任务 $k$ 的权重向量：
>
> $$
> w^k(x)=\operatorname{Softmax}\left(W_g^k x\right)
> $$
> 其中 $W_g^k \in R^{\left(m_k+m_s\right) \times d}$ 是一个参数矩阵，$m_s$ 和 $m_k$ 分别是共享专家和任务 $k$ 的特定专家的数量，$d$ 是输入表示的维度。$S^k (x)$ 是一个由所有选定向量（包括共享专家和任务 $k$ 的特定专家）组成的选定矩阵：
>
> $$
> S^k(x)=\left[E_{(k, 1)}^T, E_{(k, 2)}^T, \ldots, E_{\left(k, m_k\right)}^T, E_{(s, 1)}^T, E_{(s, 2)}^T, \ldots, E_{\left(s, m_s\right)}^T\right]^T
> $$
> 
>
> 最后，任务 $k$ 的预测是：
>
> $$
> y^k(x)=t^k\left(g^k(x)\right)
> $$
> 其中 $t^k$ 表示任务 $k$ 的塔式网络。
>
> 与MMOE相比，CGC移除了任务塔式网络与其他任务的特定于任务的专家之间的连接，使不同类型的专家能够专注于有效地学习不同的知识而不会相互干扰。结合门控网络根据输入动态融合表示的优势，CGC实现了任务之间更灵活的平衡，并更好地处理了任务冲突和样本依赖的相关性。

#### 4.2 Progressive Layered Extraction

CGC separates task-specific and shared components explicitly. However, learning needs to shape out deeper and deeper semantic representations gradually in deep MTL, while normally it is not crystally clear whether the intermediate representations should be treated as shared or task-specific. To address this issue, we generalize CGC with Progressive Layered Extraction (PLE). As depicted in Fig. 5, there are multi-level extraction networks in PLE to extract higherlevel shared information. Besides gates for task-specific experts, the extraction network also employs a gating network for shared experts to combine knowledge from all experts in this layer. Thus parameters of different tasks in PLE are not fully separated in the early layer as CGC but are separated progressively in upper layers. The gating networks in higher-level extraction network take the fusion results of gates in lower-level extraction network as the selector instead of the raw input, as it may provide better information for selecting abstract knowledge extracted in higher-level experts.

> CGC明确地将特定于任务和共享组件分开。然而，在深度多任务学习（MTL）中，学习需要逐渐塑造出越来越深的语义表示，而通常并不十分清楚中间表示应该被视为共享的还是特定于任务的。为了解决这个问题，我们将CGC与渐进式分层提取（PLE）相结合进行了推广。如图5所示，PLE中有多级提取网络来提取更高级别的共享信息。除了特定于任务的专家的门控之外，提取网络还为共享专家采用了一个门控网络，以结合该层中所有专家的知识。因此，与CGC不同，PLE中不同任务的参数在早期层中并未完全分离，而是在上层中逐步分离。高级提取网络中的门控网络将低级提取网络中门的融合结果作为选择器，而不是原始输入，因为它可以为选择高级专家提取的抽象知识提供更好的信息。

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure5.png)

The calculation of weighting function, selected matrix, and gating network in PLE are the same as that in CGC. Specifically, the formulation of the gating network of task $k$ in the $j$ th extraction network of PLE is:
$$
g^{k, j}(x)=w^{k, j}\left(g^{k, j-1}(x)\right) S^{k, j}(x)
$$
where $w^{k,j}$ is the weighting function of task $k$ with $g^{k,j−1}$ as input, and $S^{k,j}$ is the selected matrix of task $k$ in the $j$th extraction network. It is worth noting that the selected matrix of the shared module in PLE is slightly different from task-specific modules, as it consists of all shared experts and task-specific experts in this layer.

After calculating all gating networks and experts, we can obtain the prediction of task $k$ in PLE finally:
$$
y^k(x)=t^k\left(g^{k, N}(x)\right)
$$
With multi-level experts and gating networks, PLE extracts and combines deeper semantic representations for each task to improve generalization. As shown in Fig. 1, the routing strategy is full connection for MMOE and early separation for CGC. Differently,

> 在PLE中，权重函数、选定矩阵和门控网络的计算与CGC中的相同。具体来说，PLE中第$j$个提取网络中任务$k$的门控网络公式为：
>
> $g^{k, j}(x)=w^{k, j}\left(g^{k, j-1}(x)\right) S^{k, j}(x)$
>
> 其中，$w^{k,j}$是任务$k$的权重函数，以$g^{k,j−1}$为输入，而$S^{k,j}$是第$j$个提取网络中任务$k$的选定矩阵。值得注意的是，PLE中共享模块的选择矩阵与特定于任务的模块略有不同，因为它由该层中的所有共享专家和特定于任务的专家组成。
>
> 在计算了所有的门控网络和专家之后，我们最终可以得到PLE中任务$k$的预测：
>
> $y^k(x)=t^k\left(g^{k, N}(x)\right)$
>
> 借助多级专家和门控网络，PLE为每个任务提取并结合了更深的语义表示，以提高泛化能力。如图1所示，MMOE的路由策略是全连接，而CGC是早期分离。不同的是，PLE采用了渐进式分离的策略。
>
> 在PLE中，通过多级提取网络，任务之间的参数并不是在一开始就完全分离，而是在上层网络中逐渐分离。这种设计允许模型在较低层级共享更多的信息，而随着层级的提升，逐渐分离出特定于任务的信息。这种方式结合了共享和特定于任务的知识，有助于模型更好地学习和泛化。每一层的门控网络都根据前一层门控网络的融合结果进行选择，这有助于模型选择更抽象的、层级更高的知识。通过这种方式，PLE能够更好地处理多任务学习中的任务冲突和样本依赖相关性问题。
>

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Figure6.png)

#### 4.3 Joint Loss Optimization for MTL

Having designed the efficient network structure, we now focus on training task-specific and shared layers jointly in an end-to-end manner. In multi-task learning, a common formulation of joint loss is the weighted sum of the losses for each individual task:
$$
L\left(\theta_1, \ldots, \theta_K, \theta_s\right)=\sum_{k=1}^K \omega_k L_k\left(\theta_k, \theta_s\right)
$$
where $θ_s$ denotes shared parameters, $K$ is the number of tasks, $L_k$ , $ω_k$ and $θ_k$ are loss function, loss weight, and task-specific parameters of task $k$​ respectively.

> 在设计了高效的网络结构之后，我们接下来将专注于以端到端的方式联合训练特定任务和共享层。在多任务学习中，联合损失的一个常见公式是每个单独任务的损失的加权和，如下所示：
>
> $$
> L\left(\theta_1, \ldots, \theta_K, \theta_s\right)=\sum_{k=1}^K \omega_k L_k\left(\theta_k, \theta_s\right)
> $$
>
> 在这个公式中，$\theta_s$ 表示共享参数，$K$ 是任务的数量，$L_k$ 是任务 $k$ 的损失函数，$\omega_k$ 是任务 $k$ 的损失权重，用于调整每个任务损失在总损失中的贡献，而 $\theta_k$ 是任务 $k$ 的特定参数。
>

However, there exist several issues, making joint optimization of MTL models challenging in practice. In this paper, we optimize the joint loss function to address two critical ones encountered in real-world recommender systems. The first problem is the heterogeneous sample space due to sequential user actions. For instance, users can only share or comment on an item after clicking it, which leads to different sample space of different tasks shown in Fig. 6. To train these tasks jointly, we consider the union of sample space of all tasks as the whole training set, and ignore samples out of its own sample space when calculating the loss of each individual task:
$$
L_k\left(\theta_k, \theta_s\right)=\frac{1}{\sum_i \delta_k^i} \sum_i \delta_k^i \operatorname{loss}_k\left(\hat{y}_k^i\left(\theta_k, \theta_s\right), y_k^i\right)
$$
where lossk is task $k$’s loss of sample $i$ calculated based on prediction $\hat{y}_k^i$ and ground truth $y_k^i$ , $\delta_k^i \in\{0,1\}$ indicates whether sample $i$ lies in the sample space of task $k$. 

> 然而，在实际操作中，多任务学习（MTL）模型的联合优化面临着几个挑战。在本文中，我们优化了联合损失函数，以解决在现实推荐系统中遇到的两个关键问题。第一个问题是由于用户顺序操作导致的异构样本空间。例如，用户只有在点击某个项目后才能进行分享或评论，这导致了如图6所示的不同任务具有不同的样本空间。为了联合训练这些任务，我们将所有任务的样本空间的并集视为整个训练集，并在计算每个单独任务的损失时忽略不属于其自身样本空间的样本：
>
> $$
> L_k\left(\theta_k, \theta_s\right)=\frac{1}{\sum_i \delta_k^i} \sum_i \delta_k^i \operatorname{loss}_k\left(\hat{y}_k^i\left(\theta_k, \theta_s\right), y_k^i\right)
> $$
>
> 在上述公式中，$\operatorname{loss}_k$ 表示任务 $k$ 中样本 $i$ 的损失，该损失是基于预测值 $\hat{y}_k^i$ 和真实值 $y_k^i$ 计算得出的。变量 $\delta_k^i \in\{0,1\}$ 是一个指示器，用于表示样本 $i$ 是否属于任务 $k$ 的样本空间。如果样本 $i$ 属于任务 $k$ 的样本空间，则 $\delta_k^i = 1$，否则 $\delta_k^i = 0$。通过这种方式，我们可以确保在计算每个任务的损失时，只考虑属于该任务样本空间的样本，从而解决异构样本空间的问题。
>
> 这种方法允许我们有效地联合训练多个任务，即使它们的样本空间不完全重叠。通过这种方式，模型可以在共享参数的同时，学习到每个任务的特定特征，从而提高整体性能和泛化能力。

The second problem is that the performance of an MTL model is sensitive to the choice of loss weight in the training process [9], as it determines the relative importance of each task on the joint loss. In practice, it is observed that each task may have different importance at different training phases. Therefore, we consider the loss weight for each task as a dynamic weight instead of a static one. At first, we set an initial loss weight $ω_{k,0}$ for task $k$, then update its loss weight after each step based on the updating ratio $γ_k$ :
$$
\omega_k^{(t)}=\omega_{k, 0} \times \gamma_k^t
$$
where $t$ denotes the training epoch, $ω_{k,0}$ and $γ_k$ are hyper-parameters of the model.

> 第二个问题是，多任务学习（MTL）模型的性能对训练过程中损失权重的选择非常敏感[9]，因为这决定了每个任务在联合损失中的相对重要性。在实践中，观察到每个任务在不同的训练阶段可能有不同的重要性。因此，我们将每个任务的损失权重视为动态权重，而不是静态权重。首先，我们为任务$k$设置一个初始损失权重$\omega_{k,0}$，然后基于更新比率$\gamma_k$在每一步之后更新其损失权重：
>
> $$
> \omega_k^{(t)}=\omega_{k, 0} \times \gamma_k^t
> $$
>
> 其中，$t$ 表示训练周期（epoch），$\omega_{k,0}$和$\gamma_k$是模型的超参数。
>
> 通过这种方式，模型可以在训练过程中动态地调整每个任务的损失权重。这有助于模型更好地适应不同任务在不同训练阶段的重要性变化，从而提高整体性能和适应性。同时，这种方法也增加了模型的灵活性和可调性，使得模型能够更好地应对复杂多变的多任务学习场景。

## 5 EXPERIMENTS

In this section, extensive offline and online experiments are performed on both the large-scale recommender system in Tencent and public benchmark datasets to evaluate the effectiveness of proposed models. We also analyze the expert utilization in all gatebased MTL models to better understand the working mechanism of gating networks and verify the structure value of CGC and PLE further.

> 在本节中，为了评估所提出模型的有效性，我们在腾讯的大规模推荐系统和公共基准数据集上进行了大量的离线和在线实验。此外，我们还深入分析了所有基于门机制的多任务学习（MTL）模型中的专家利用率，以便更好地理解门控网络的工作机制，并进一步验证CGC（Customized Gate Control）和PLE（Progressive Layered Extraction）的结构价值。

#### 5.1 Evaluation on the Video Recommender System in Tencent

In this subsection, we conduct offline and online experiments for task groups with complex and normal correlations as well as multiple tasks in the video recommender system at Tencent to evaluate the performance of proposed models.

> 在本小节中，我们对腾讯视频推荐系统中具有复杂和正常相关性的任务组以及多任务进行了离线和在线实验，以评估所提出模型的性能。

*5.1.1 Dataset.* We collect an industrial dataset through sampling user logs from the video recommender system serving Tencent News during 8 consecutive days. There are 46.926 million users,2.682 million videos and 0.995 billion samples in the dataset. As mentioned before, VCR, CTR, VTR, SHR (Share Rate), and CMR (Comment Rate) are tasks modeling user preferences in the dataset.

*5.1.2 Baseline Models.* In the experiment, we compare CGC and PLE with single-task model, asymmetric sharing, customized sharing, and the SOTA MTL models including cross-stitch network, sluice network, and MMOE. As multi-level experts are shared in PLE, we extend MMOE to ML-MMOE (multi-layer MMOE) shown in Fig. 1h) by adding multi-level experts for fair comparison. In ML-MMOE, higher-level experts combine representations from lower-level experts through gating networks and all gating networks share the same selector.

*5.1.3 Experiment Setup.* In the experiment, VCR prediction is a regression task trained and evaluated with MSE loss, tasks modeling other actions are all binary classification tasks trained with crossentropy loss and evaluated with AUC. Samples in the first 7 days are used for training and the rest samples are test set. We adopt a three-layer MLP network with RELU activation and hidden layer size of [256, 128, 64] for each task in both MTL models and the single-task model. For MTL models, we implement the expert as a single-layer network and tune the following model-specific hyperparameters: number of shared layers, cross-stitch units in hard parameter sharing and cross-stitch network, number of experts in all gate-based models. For fair comparison, we implement all multi-level MTL models as two-level models to keep the same depth of networks.

> **5.1.1 数据集**：我们从腾讯视频推荐系统中连续8天的用户日志中采样，收集了一个工业数据集。该数据集包含4692.6万用户、268.2万视频和9.95亿样本。如前所述，VCR（视频完成率）、CTR（点击率）、VTR（视频观看率）、SHR（分享率）和CMR（评论率）是数据集中建模用户偏好的任务。
>
> **5.1.2 基准模型**：在实验中，我们将CGC和PLE与单任务模型、非对称共享、定制共享以及最先进的多任务学习（MTL）模型进行比较，包括cross-stitch网络、sluice网络和MMOE。由于PLE中共享了多级专家，我们将MMOE扩展到ML-MMOE（多层MMOE）（如图1h所示），通过添加多级专家以进行公平比较。在ML-MMOE中，更高级别的专家通过门控网络结合来自较低级别专家的表示，并且所有门控网络共享相同的选择器。
>
> **5.1.3 实验设置**：在实验中，VCR预测是一个回归任务，使用均方误差（MSE）损失进行训练和评估；建模其他动作的任务都是二元分类任务，使用交叉熵损失进行训练，并使用AUC进行评估。前7天的样本用于训练，其余的样本作为测试集。我们在多任务学习模型和单任务模型中为每个任务采用了一个具有RELU激活和隐藏层大小为[256, 128, 64]的三层多层感知器（MLP）网络。对于多任务学习模型，我们将专家实现为单层网络，并调整以下模型特定的超参数：共享层的数量、硬参数共享和cross-stitch网络中的cross-stitch单元数量、以及所有基于门的模型中的专家数量。为了公平比较，我们将所有多级多任务学习模型实现为两级模型，以保持网络深度相同。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Table1.png)

Besides common evaluation metrics such as AUC and MSE, we define a metric of MTL gain to quantitively evaluate the benefit of multi-task learning over the single-task model for a certain task. As shown in Equation 11, for a given task group and an MTL model q, MTL gain of q on task A is defined as the task A’s performance improvement of MTL model q over the single-task model with the same network structures and training samples.

> 除了常见的评估指标，如AUC和MSE之外，我们还定义了一个多任务学习增益指标，以定量评估多任务学习相对于单任务模型在特定任务上的优势。如等式11所示，对于给定的任务组和多任务学习模型q，q在任务A上的多任务学习增益定义为多任务学习模型q相对于具有相同网络结构和训练样本的单任务模型在任务A上的性能提升。

$$
\text { MTL gain }= \begin{cases}M_{M T L}-M_{\text {single }}, & M \text { is a positive metric } \\ M_{\text {single }}-M_{M T L}, & M \text { is a negative metric }\end{cases}
$$

*5.1.4 Evaluation on Tasks with Complex Correlation.* To better capture the major online engagement metrics, e.g., view count and watch time, we first conduct experiments on task-group of VCR/VTR. Table 1 illustrates the experiment results and we mark best scores in bold and performance degeneration (negative MTL gain) in gray. It is shown that CGC and PLE significantly outperform all baseline models in VTR. Due to the complex correlation between VTR and VCR, we can clearly observe the seesaw phenomenon with the zigzag gray distribution that some models improve VCR but hurt VTR while some improve VTR but hurt VCR. Specifically, MMOE improves both tasks over single-task but the improvement is not significant, while ML-MMOE improves VTR but hurts VCR. Compared to MMOE and ML-MMOE, CGC improves VTR much more and improves VCR slightly as well. Finally, PLE converges with similar pace and achieves significant improvement over the above models with the best VCR MSE and one of the best VTR AUCs.

> **5.1.4 对具有复杂相关性的任务进行评估**：为了更好地捕捉主要的在线参与度指标，例如观看次数和观看时间，我们首先进行了 VCR（视频完成率）/ VTR（视频观看率）任务组的实验。表1展示了实验结果，我们用粗体标记了最佳分数，用灰色标记了性能退化（多任务学习增益为负）的情况。结果显示，CGC和PLE在VTR方面显著优于所有基准模型。由于VTR和VCR之间存在复杂的相关性，我们可以清楚地观察到一些模型在提高VCR时会损害VTR，而在提高VTR时又会损害VCR的跷跷板现象，这种现象在灰色分布中呈现出曲折的特点。具体来说，MMOE相对于单任务模型在两个任务上都有所提高，但提高并不显著，而ML-MMOE则提高了VTR但损害了VCR。与MMOE和ML-MMOE相比，CGC在VTR上的提高更为显著，同时在VCR上也有略微提高。最终，PLE以相似的速度收敛，并在上述模型中取得了显著的改进，具有最佳的VCR MSE和最好的VTR AUC之一。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Table2.png)

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Table3.png)

5.1.6 Online A/B Testing. Careful online A/B test with task-group of VTR and VCR was conducted in the video recommender system for 4 weeks. We implement all MTL models in our C++ based deep learning framework, randomly distribute users into several buckets, and deploy each model to one of the bucket. The final ranking score is obtained through the combination function of multiple predicted scores described in Section 3. Table 3 shows the improvement of MTL models over the single-task model on online metrics of total view count per user and total watch time per user, the ultimate goal of the system. It is shown that CGC and PLE achieve significant increase in online metrics over all baseline models. Moreover, PLE outperforms CGC significantly on all online metrics, which shows that small improvements of AUC or MSE in MTL yield significant improvements in online metrics. PLE has been deployed to the platform in Tencent since then.

> 5.1.6 在线A/B测试。我们在视频推荐系统中对VTR和VCR任务组进行了为期4周的仔细在线A/B测试。我们在基于C++的深度学习框架中实现了所有的多任务学习模型，将用户随机分配到几个组中，并将每个模型部署到一个组中。最终排名得分是通过第3节中描述的多个预测得分的组合函数获得的。表3显示了多任务学习模型相对于单任务模型在在线指标上的改进，包括每个用户的总观看次数和每个用户的总观看时间，这是系统的最终目标。结果表明，CGC和PLE在所有基准模型的在线指标上都取得了显著提升。此外，PLE在所有在线指标上都明显优于CGC，这表明多任务学习中AUC或MSE的微小改进可以显著提高在线指标。自那以后，PLE已被部署到腾讯的平台上。

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/multitask/PLE/Table4.png)

> 5.1.7 多任务评估。最后，我们探索了CGC和PLE在更具挑战性的多任务场景中的可扩展性。除了VTR（视频观看率）和VCR（视频完成率）之外，我们还引入了SHR（分享率）和CMR（评论率）来模拟用户反馈行为。将CGC和PLE扩展到多任务情况是灵活的，只需为每个任务添加一个任务特定的专家模块、门控网络和塔式网络。如表4所示，CGC和PLE几乎在所有任务组的所有任务上都显著优于单任务模型。这表明，对于包含两个以上任务的一般情况，CGC和PLE仍然展现出促进任务合作、防止负迁移和跷跷板现象的优势。在所有情况下，PLE都明显优于CGC。因此，PLE在提高不同任务组规模的共享学习效率方面表现出更强的优势。

#### 5.2 Evaluation on Public Datasets

In this subsection, we conduct experiments on public benchmark datasets to evaluate the effectiveness of PLE in scenarios besides recommendation further.

**5.2.1 Datasets. **

- Synthetic Data is generated following the data synthesizing process based on [13] to control task correlations. As hyperparameters for data synthesis are not provided in [13], we randomly sample αi and βi following the standard normal distribution, and set c=1, m=10, d=512 for reproducibility. 1.4 million samples with two continuous labels are generated for each correlation.
- Census-income Dataset [5] contains 299,285 samples and 40 features extracted from the 1994 census database. For fair comparison with baseline models, we consider the same task-group as [13]. In detail, task 1 aims to predict whether the income exceeds 50K, task 2 aims to predict whether this person’s marital status is never married. 
- Ali-CCP Dataset 1 is a public dataset containing 84 million samples extracted from Taobao’s Recommender System. CTR and CVR (Conversion Rate) are two tasks modeling actions of click and purchase in the dataset

> 在本小节中，我们在公共基准数据集上进行实验，以进一步评估PLE在推荐场景之外的有效性。
>
>  * **合成数据**：根据[13]的数据合成过程生成合成数据，以控制任务之间的相关性。由于[13]中没有提供数据合成的超参数，我们按照标准正态分布随机抽样αi和βi，并设置c=1，m=10，d=512以确保可重复性。针对每种相关性，生成了带有两个连续标签的140万个样本。
>
> * **Census-income（人口普查收入）数据集**：[5]包含从1994年人口普查数据库中提取的299,285个样本和40个特征。为了与基准模型进行公平比较，我们考虑了与[13]相同的任务组。具体来说，任务1旨在预测收入是否超过50K，任务2旨在预测此人的婚姻状况是否为未婚。
>
> * **Ali-CCP数据集1**：这是一个公开数据集，包含从淘宝推荐系统中提取的8400万个样本。数据集中的CTR（点击率）和CVR（转化率）是两个建模点击和购买动作的任务。

**5.2.2 Experiment Setup.** The setup for census-income dataset is the same as [13]. For synthetic data and Ali-CCP dataset, we adopt a three-layer MLP network with RELU activation and hidden layer size of [256, 128, 64] for each task in both MTL models and singletask model. Hyper-parameters are tuned similarly to the experiments in Section 5.1.

**5.2.3 Experiment Results.** 8 Experiment results on synthetic data shown in Fig. 7 demonstrate that hard parameter sharing and MMOE sometimes suffer from seesaw phenomenon and lost balance between two tasks. On the contrary, PLE consistently performs best for both tasks across different correlations and achieves 87.2% increase in MTL gain over MMOE on average. As results on AliCCP and census-income dataset shown in Table 5, PLE eliminates the seesaw phenomenon and outperforms the single-task model and MMOE consistently on both tasks.

Combined with previous experiments on the industrial dataset and online A/B test, PLE exhibits stable general benefit on improving MTL efficiency and performance for different task correlation patterns and different applications.

> **5.2.2 实验设置**：人口普查收入数据集的设置与[13]中的设置相同。对于合成数据和Ali-CCP数据集，我们在多任务学习模型和单任务模型中为每个任务都采用了具有RELU激活函数和隐藏层大小为[256, 128, 64]的三层MLP网络。超参数的调整与5.1节中的实验类似。
>
> **5.2.3 实验结果**：图7中显示的合成数据上的实验结果表明，硬参数共享和MMOE有时会受到跷跷板现象的影响，并在两个任务之间失去平衡。相反，PLE在不同相关性下的两个任务中都表现最佳，并且在多任务学习增益上平均比MMOE提高了87.2%。如表5所示，在AliCCP和人口普查收入数据集上的结果，PLE消除了跷跷板现象，在两个任务上都始终优于单任务模型和MMOE。
>
> 结合之前在工业数据集和在线A/B测试上的实验，PLE在提高多任务学习效率和性能方面展现出了稳定的普遍优势，适用于不同的任务相关性模式和应用场景。这表明PLE模型在处理多任务学习问题时，能够有效地平衡各个任务的性能，避免跷跷板现象，从而实现整体性能的提升。这些实验结果验证了PLE模型的有效性和适用性。

#### 5.3 Expert Utilization Analysis

To disclose how the experts are aggregated by different gates, we investigate expert utilization of all gate-based models in VTR/VCR task group of the industrial dataset. For simplicity and fair comparison, we consider each expert as a single-layer network, keep only one expert in each expert module of CGC and PLE, while keep three experts in each layer of MMOE and ML-MMOE. Fig. 8 shows the weight distribution of experts utilized by each gate in all testing data, where the height of bars and vertical short lines indicate mean and standard deviation of weights respectively. It is shown that the VTR and VCR combine experts with significantly different weights in CGC while much similar weights in MMOE, which indicates that the well-designed structure of CGC helps achieve better differentiation between different experts. Furthermore, there is no zero-weight for all experts in MMOE and ML-MMOE, which further shows that it is hard for MMOE and ML-MMOE to converge to the structure of CGC and PLE without prior knowledge in practice, despite the existence of theoretical possibility. Compared with CGC, shared experts in PLE have larger influence on the input of tower networks especially for the VTR task. The fact that PLE performs better than CGC shows the value of shared higher-level deeper representations. In other words, it is demanded that certain deeper semantic representations are shared between tasks thus a progressive separation routing provides a better joint routing and learning scheme.

> 为了揭示不同的门是如何聚合专家的，我们调查了工业数据集中VTR/VCR任务组所有基于门的模型的专家利用率。为了简化和公平比较，我们将每个专家视为单层网络，在CGC和PLE的每个专家模块中只保留一个专家，而在MMOE和ML-MMOE的每一层中保留三个专家。图8显示了所有测试数据中每个门利用的专家的权重分布，其中条形的高度和垂直短线分别表示权重的均值和标准差。
>
> 结果表明，在CGC中，VTR和VCR以显著不同的权重组合专家，而在MMOE中，专家的权重则非常相似，这表明CGC的良好设计结构有助于更好地实现不同专家之间的差异化。此外，MMOE和ML-MMOE中的所有专家都没有零权重，这进一步表明，在实践中，如果没有先验知识，MMOE和ML-MMOE很难收敛到CGC和PLE的结构，尽管存在理论上的可能性。
>
> 与CGC相比，PLE中的共享专家对塔式网络的输入有更大的影响，尤其是对VTR任务。PLE比CGC表现更好的事实表明了共享更高级别更深层次表示的价值。换句话说，需要在任务之间共享某些更深层次的语义表示，因此，渐进式分离路由提供了更好的联合路由和学习方案。
>
> 这段文字深入探讨了不同多任务学习模型（如CGC、PLE、MMOE和ML-MMOE）中专家权重的分配和利用情况，通过对比实验揭示了这些模型在任务间的表示共享和差异化方面的特点和性能。实验结果表明，具有良好设计的结构（如CGC和PLE）能够更好地在任务间实现表示的共享和差异化，从而提升多任务学习的效果。

## 6 CONCLUSION

In this paper, we propose a novel MTL model called Progressive Layered Extraction (PLE), which separates task-sharing and taskspecific parameters explicitly and introduces an innovative progressive routing manner to avoid the negative transfer and seesaw phenomenon, and achieve more efficient information sharing and joint representation learning. Offline and online experiment results on the industrial dataset and public benchmark datasets show significant and consistent improvements of PLE over SOTA MTL models. Exploring the hierarchical task-group correlations will be the focus of future work.

> 在本文中，我们提出了一种新颖的多任务学习（MTL）模型，称为逐层提取（PLE）。该模型明确地区分了任务共享参数和任务特定参数，并引入了一种创新的渐进式路由方式，以避免负迁移和跷跷板现象，并实现更有效的信息共享和联合表示学习。在工业数据集和公共基准数据集上的离线和在线实验结果表明，与最先进的多任务学习模型相比，PLE具有显著且一致的改进。探索层次化的任务组相关性将是未来工作的重点。