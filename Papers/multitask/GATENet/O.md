# GateNet:Gating-Enhanced Deep Network for Click-Through Rate Prediction



## ABSTRACT

Advertising and feed ranking are essential to many Internet companies such as Facebook. Among many real-world advertising and feed ranking systems, click through rate (CTR) prediction plays a central role. In recent years, many neural network based CTR models have been proposed and achieved success such as Factorization-Machine Supported Neural Networks, DeepFM and xDeepFM. Many of them contain two commonly used components: embedding layer and MLP hidden layers. On the other side, gating mechanism is also widely applied in many research fields such as computer vision(CV) and natural language processing(NLP). Some research has proved that gating mechanism improves the trainability of non-convex deep neural networks. Inspired by these observations, we propose a novel model named GateNet which introduces either the feature embedding gate or the hidden gate to the embedding layer or hidden layers of DNN CTR models, respectively. The feature embedding gate provides a learnable feature gating module to select salient latent information from the feature-level. The hidden gate helps the model to implicitly capture the high-order interaction more effectively. Extensive experiments conducted on three real-world datasets demonstrate its effectiveness to boost the performance of various state-of-the-art models such as FM, DeepFM and xDeepFM on all datasets.

> 广告和Feed流排名对于许多互联网公司（如Facebook）来说至关重要。在现实世界的广告和Feed流排名系统中，点击率（CTR）预测发挥着核心作用。近年来，已经有许多基于神经网络的CTR模型被提出并取得了成功，如因子分解机支持的神经网络（Factorization-Machine Supported Neural Networks）、DeepFM和xDeepFM。这些模型中许多都包含两个常用的组件：嵌入层和多层感知器（MLP）隐藏层。另一方面，门控机制在计算机视觉（CV）和自然语言处理（NLP）等许多研究领域也得到了广泛应用。一些研究表明，门控机制提高了非凸深度神经网络的可训练性。受这些观察结果的启发，我们提出了一种名为GateNet的新型模型，该模型将特征嵌入门或隐藏门分别引入到DNN CTR模型的嵌入层或隐藏层中。特征嵌入门提供了一个可学习的特征门控模块，以从特征级别中选择显著的潜在信息。隐藏门则帮助模型更有效地隐式捕获高阶交互。在三个真实数据集上进行的广泛实验表明，它可以有效提高各种最新模型（如FM、DeepFM和xDeepFM）在所有数据集上的性能。

## 1 INTRODUCTION

Advertising and feed ranking are essential to many Internet companies such as Facebook. The main technique behind these tasks is click-through rate prediction which is known as CTR. Many models have been proposed in this field such as logistic regression (LR)[19], polynomial-2 (Poly2)[12], tree based models[10], tensor- based models[14], Bayesian models[8], and factorization machines based models[12, 21].

> 广告和Feed流排名对于Facebook等许多互联网公司来说至关重要。这些任务背后的主要技术是点击率预测，即CTR。在这个领域已经提出了许多模型，如逻辑回归（LR）[19]、二次多项式模型（Poly2）[12]、基于树的模型[10]、基于张量的模型[14]、贝叶斯模型[8]和基于分解机的模型[12, 21]。

With the great success of deep learning in many research fields such as computer vision[15] and natural language processing[2, 20], many deep learning based CTR models have been proposed in recent years[1, 9, 16, 25? ]. Many of them contain two commonly used components:embedding layer and MLP hidden layers. On the other side, gating mechanism is also widely applied in many research fields such as computer vision(CV) and natural language processing(NLP). Some research works have proved that gating mechanism improves the trainability of non-convex deep neural networks. Inspired by these observations, a model named GateNet is proposed to select salient latent information from the feature-level and implicitly capture the high-order interaction more effectively for CTR prediction.

> 随着深度学习在计算机视觉[15]和自然语言处理[2, 20]等许多研究领域取得巨大成功，近年来提出了许多基于深度学习的CTR模型[1, 9, 16, 25? ]。这些模型中许多都包含两个常用组件：嵌入层和多层感知器（MLP）隐藏层。另一方面，门控机制也在计算机视觉（CV）和自然语言处理（NLP）等许多研究领域得到广泛应用。一些研究工作已经证明，门控机制提高了非凸深度神经网络的可训练性。受这些观察的启发，我们提出了一种名为GateNet的模型，该模型从特征层面选择显著的潜在信息，并更有效地隐式捕获高阶交互，以进行CTR预测。

Our main contributions are listed as follows:

- We propose the feature embedding gate layer to replace the traditional embedding and enhance the model ability. Inserting the feature embedding gate into the embedding layer of many classical models such as FM, DeepFM, DNN and XDeepFM, we observe a significant performance improve- ment.
- The MLP layers are an essential component to implicitly capturing the high-order feature interaction in the canonical DNN models, we introduce the hidden gate to the MLP parts of deep models and improve the performance of the the classical models.
- ItissimpleandeffectivetoenhancethestandardDNNmodel by inserting hidden gate and we can achieve comparable per- formance with other state-of-the-art model baselines such as DeepFM and XDeepFM.

The rest of this paper is organized as follows. In Section 2, we review related works which are relevant with our proposed model, followed by introducing our proposed model in Section 3. We will present experimental explorations on three real-world datasets in Section 4. Finally, we conclude this work in Section 5.

> 我们的主要贡献列举如下：
>
> - 我们提出了特征嵌入门层来替代传统的嵌入层，以增强模型能力。将特征嵌入门插入到许多经典模型（如FM、DeepFM、DNN和XDeepFM）的嵌入层中，我们观察到性能有了显著提升。
> - 在标准的DNN模型中，MLP层是隐式捕获高阶特征交互的重要组成部分，我们将隐藏门引入到深度模型的MLP部分，提高了经典模型的性能。
> - 通过插入隐藏门来增强标准DNN模型既简单又有效，我们可以达到与其他最先进的模型基线（如DeepFM和XDeepFM）相当的性能。
>
> 本文的其余部分组织如下。在第2节中，我们将回顾与我们提出的模型相关的研究工作，然后在第3节中介绍我们提出的模型。我们将在第4节中对三个真实数据集进行实验探索。最后，我们在第5节中总结这项工作。

## 2 RELATED WORK

#### 2.1 Deep Learning based CTR Models

Many deep learning based CTR models have also been proposed in recent years[1, 9, 16, 24, 25]. How to effectively model the fea- ture interactions is the key factor for most of these neural network based models. Factorization-Machine Supported Neural Networks (FNN)[25] is a forward neural network using FM to pre-train the embedding layer. However, FNN can capture only high-order fea- ture interactions. Wide & Deep model(WDL)[1] jointly trains wide linear models and deep neural networks to combine the benefits of memorization and generalization for recommendation systems. However, expertise feature engineering is still needed on the input to the wide part of WDL. To alleviate manual efforts in feature en- gineering, DeepFM[9] replaces the wide part of WDL with FM and shares the feature embedding between the FM and deep component.

In addition, Deep & Cross Network (DCN)[24] and eXtreme Deep Factorization Machine (xDeepFM)[16] are recent deep learning methods which explicitly model the feature interactions.

> 近年来也提出了许多基于深度学习的CTR模型[1, 9, 16, 24, 25]。如何有效地对特征交互进行建模是这些基于神经网络的模型中的关键因素。因子分解机支持的神经网络（FNN）[25]是一种前馈神经网络，它使用因子分解机（FM）来预训练嵌入层。然而，FNN只能捕获高阶特征交互。Wide & Deep模型（WDL）[1]联合训练了Wide线性模型和Deep神经网络，以结合推荐系统的记忆和泛化能力。但是，对于输入到WDL的Wide部分，仍然需要进行专业的特征工程。为了减少特征工程中的手动工作，DeepFM[9]使用FM替换了WDL中的Wide部分，并在FM和深度组件之间共享了特征嵌入。
>
> 此外，深度与交叉网络（DCN）[24]和极深因子分解机（xDeepFM）[16]是最近提出的深度学习方法，它们可以显式地对特征交互进行建模。

#### 2.2 Gating Mechanisms in Deep Learning

Gating mechanism is widely used in many deep learning fields, such as computer vision(CV), natural language processing(NLP), and recommendation systems.

The gate mechanism is used in computer vision, such as Highway Network [23], they utilize the transform gate and the carry gate to express how much of the output is produced by transforming the input and carrying the output, respectively.

The gate mechanism is widely applied to NLP, such as LSTM[6], GRU[2], language modeling[4], sequence to sequence learning[5] and they utilize the gate to prevent the gradients vanishing and resolve the long-term dependency problem.

In addition, [18] uses the gates to automatically adjust parameters between modeling shared information and modeling task- specific information in recommendation systems. Another recommendation system applying the gate mechanism is hierarchical gating network(HGN)[17] and they apply feature-level and instance- level gating modules to adaptively control what item latent features and which relevant item can be passed to the downstream layers.

> 门控机制在许多深度学习领域都有广泛应用，如计算机视觉（CV）、自然语言处理（NLP）和推荐系统。
>
> 在计算机视觉中，比如Highway Network[23]就使用了门控机制。它们利用变换门和传输门来表达输出中有多少是由变换输入和传输输出产生的。
>
> 门控机制在自然语言处理中也有广泛应用，如LSTM[6]、GRU[2]、语言建模[4]、序列到序列学习[5]等。它们利用门控机制来防止梯度消失并解决长期依赖问题。
>
> 此外，[18]使用门控机制在推荐系统中自动调整建模共享信息和建模任务特定信息之间的参数。另一个应用门控机制的推荐系统是分层门控网络（HGN）[17]，它们应用特征级别和实例级别的门控模块来自适应地控制哪些项目的潜在特征和哪些相关项目可以传递到下游层。
>

## 3 OUR PROPOSED MODEL

Deep learning models are widely used in industrial recommendation systems, such as WDL, YouTubeNet[3] and DeepFM. The DNN model is a sub-component in many current DNN ranking systems, and its network structure is shown in the left of Figure 1.

We can find two commonly used components in most of the current DNN ranking systems: the embedding layer and MLP hidden layer. We aim to enhance the model ability and propose the model named GateNet for CTR prediction tasks. First, we propose the feature embedding gating layer which can convert embedding features into gate-aware embedding features and helps to select salient latent information from the feature-level. Second, we also propose the hidden gate which can adaptively control what latent features and which relevant feature interaction can be passed to the downstream layer. The DNN model with feature embedding gate and DNN model with hidden gate are depicted as the middle and right in Figure 1. In the following subsections, we will describe the feature embedding layer and hidden gate layer in GateNet in detail.

> 深度学习模型在工业推荐系统中得到广泛应用，如WDL、YouTubeNet[3]和DeepFM。DNN模型是许多当前DNN排序系统中的一个子组件，其网络结构如图1左侧所示。
>
> 在大多数当前的DNN排序系统中，我们可以发现两个常用的组件：嵌入层和MLP隐藏层。我们的目标是提高模型能力，并针对点击率（CTR）预测任务提出了名为GateNet的模型。首先，我们提出了特征嵌入门控层，它可以将嵌入特征转换为具有门控意识的嵌入特征(gate-aware embedding features)，并有助于从特征级别选择显著的潜在信息。其次，我们还提出了隐藏门，它可以自适应地控制哪些潜在特征和哪些相关的特征交互可以传递到下游层。带有特征嵌入门的DNN模型和带有隐藏门的DNN模型分别如图1中间和右侧所示。在以下小节中，我们将详细描述GateNet中的特征嵌入层和隐藏门层。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Figure1.png)

#### 3.1 Feature Embedding Gate

The sparse input layer and embedding layer are widely used in deep learning based CTR models such as DeepFM[9]. The sparse input layer adopts a sparse representation for raw input features.

The embedding layer is able to embed the sparse feature into a low dimensional, dense real-value vector. The output of embedding layer is a wide concatenated field embedding vector:
$$
E=\left[e_1, e_2, \cdots, e_i, \cdots, e_f\right]
$$
where $f$ denotes the number of fields, $e_i \in R^k$ denotes the embedding of $i$-th field, and $k$ is the dimension of embedding layer.

On the other side, recent research results show that gate can improve the train-ability in training non-convex deep neural networks[7]. In this work, firstly we propose the feature embedding gate to select salient latent information from the feature-level in the DeepCTR model. The basic steps of the feature embedding gate can be described as followed:

First, for every field embedding $e_i$ , we calculate the gate value which represents the feature-level importance of embedding. We formalize this step as the following formula:
$$
g_i=\sigma\left(W_i \cdot e_i\right)
$$
where $\sigma$ is the activation function of gate, $e_i \in R^k$ is the original embedding, $W_i$ is the learned parameters of the $i$-th gate and the total number of learned parameter matrix $W=\left[W_1, \cdots, W_i, \cdots, W_f\right]$, $i=1, \cdots, f$.

Second, we assign the gate value to the corresponding feature embedding and generate a gate-aware embedding.
$$
g e_i=e_i \odot g_i
$$
where $⊙$ denotes the Hadamard or element-wise product, $e_i \in R^k$ is the $i$-th original embedding, $i =1,··· ,f$.

Third, we collect all gate-aware embeddings and regard it as gated feature embedding.
$$
G E=\left[g e_1, g e_2, \cdots, g e_i, \cdots, g e_f\right]
$$
It is a common practice to make gate output a scalar which represents the importance of the whole feature embedding. To learn the bit level salient important information in the feature embedding, we can make this gate output a vector which contains fine-grained information about the feature embedding. And we call this embedding gate ‘bit-wise’ gate and the common gate ‘vector- wise’ gate. The vector-wise and bit-wise feature embedding gate can be depicted as Figure 2.

> 嵌入层的输出是一个广泛连接的字段嵌入向量：
>
> $$
> E=\left[e_1, e_2, \cdots, e_i, \cdots, e_f\right]
> $$
>
> 其中$f$表示字段的数量，$e_i \in R^k$表示第i个字段的嵌入，$k$是嵌入层的维度。
>
> 另一方面，最近的研究结果表明，门控机制可以提高训练非凸深度神经网络的可训练性[7]。在这项工作中，我们首先提出了特征嵌入门，以从DeepCTR模型的特征级别中选择显著的潜在信息。特征嵌入门的基本步骤可以描述如下：
>
> 首先，对于每个字段嵌入$e_i$，我们计算门控值，该值表示嵌入的特征级别重要性。我们将此步骤形式化为以下公式：
>
> $$
> g_i=\sigma\left(W_i \cdot e_i\right)
> $$
>
> 其中$\sigma$是门的激活函数，$e_i \in R^k$是原始嵌入，$W_i$是第 $i$ 个门的学习参数，学习参数矩阵的总数为$W=\left[W_1, \cdots, W_i, \cdots, W_f\right]$，$i=1, \cdots, f$。
>
> 其次，我们将门控值分配给相应的特征嵌入，并生成一个门控感知的嵌入。
>
> $$
> ge_i=e_i \odot g_i
> $$
>
> 其中$⊙$表示Hadamard或元素乘积，$e_i \in R^k$是第 $i$ 个原始嵌入，$i =1,··· ,f$。
>
> 第三，我们收集所有门控感知的嵌入，并将其视为门控特征嵌入。
>
> $$
> GE=\left[ge_1, ge_2, \cdots, ge_i, \cdots, ge_f\right]
> $$
>
> 使门输出一个标量来表示整个特征嵌入的重要性是一种常见的做法。为了学习特征嵌入中的位级显著重要信息，我们可以使这个门输出一个包含有关特征嵌入的细粒度信息的向量。我们称这种嵌入门为“位级”门，而普通门为“向量级”门。向量级和位级的特征嵌入门可以如图2所示。
>
> **理解**：
>
> 这篇文章主要介绍了在深度学习CTR模型中应用稀疏输入层和嵌入层的方法，并重点介绍了一种新的技术——特征嵌入门。这项技术旨在从特征级别中选择并强调重要的潜在信息。文章中详细描述了特征嵌入门的操作步骤，包括如何计算每个特征嵌入的门控值，如何将这些门控值应用于原始嵌入以生成门控感知的嵌入，并最终如何将这些嵌入组合起来形成门控特征嵌入。此外，文章还提到了门控机制可以输出表示整个特征嵌入重要性的标量，或者输出包含细粒度信息的向量，以更精细地表示特征嵌入中的重要性。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Figure2.png)

Seen from the figure, we compare the difference of vector-wise feature gate and bit-wise feature is as follows:
$$
\text{Vector-wise}: g_i \in R, W_i \in R^{k \times 1}, W \in R^{f \times k \times 1}.
\\
\text{Bit-wise}: g_i \in R^k, W_i \in R^{k \times k}, W \in R^{f \times k \times k}.
$$
We can see that the output of bit-wise gate is a vector which is related to each bit of feature embedding and it can be regarded as using the same value to each bit of feature embedding. The performance comparison of vector-wise and bit-wise feature embedding gate will be discussed in Section 4.2.

Moreover, as some previous works such as FiBiNet[11] does, we will explore the parameter sharing mechanism of the feature embedding gate layer. Each gate in the feature embedding gate layer has its own parameters to explicitly learn the salient feature information, we also can make all the gates share parameters in order to reduce the number of parameters. We call this gate ‘field sharing’ and previous gate ‘field private’. From a mathematical perspective, the biggest difference between ‘field sharing’ and ‘field private’ is the learned gate parameters Wi . Wi is shared among all the fields in ‘field sharing’ while Wi is different for each field in ‘field private’. The performance of ‘field sharing’ and ‘field private’ will be compared in Section 4.2.

> **翻译**：
>
> 从图中我们可以看出，向量级特征门和位级特征门的区别如下：
>
> $$
> \text{向量级}: g_i \in R, W_i \in R^{k \times 1}, W \in R^{f \times k \times 1}.
> \\
> \text{位级}: g_i \in R^k, W_i \in R^{k \times k}, W \in R^{f \times k \times k}.
> $$
>
> 我们可以看到，位级门的输出是一个与特征嵌入的每一位都相关的向量，这可以看作是对特征嵌入的每一位都使用了相同的值。向量级和位级特征嵌入门的性能比较将在4.2节中讨论。
>
> 此外，与FiBiNet[11]等一些先前的工作一样，我们将探索特征嵌入门层的参数共享机制。特征嵌入门层中的每个门都有自己的参数，以明确学习显著的特征信息，我们也可以让所有门共享参数，以减少参数数量。我们称这种门为“字段共享”，而之前的门为“字段私有”。从数学的角度来看，“字段共享”和“字段私有”之间的最大区别是学习的门参数 $W_i$。在“字段共享”中，$W_i$ 在所有字段之间共享，而在“字段私有”中，每个字段的$W_i$都是不同的。“字段共享”和“字段私有”的性能将在4.2节中进行比较。
>
> **解释**：
>
> 这部分内容主要介绍了两种特征门：向量级和位级，并解释了它们之间的主要区别。向量级特征门输出一个标量值，代表整个特征嵌入的重要性，而位级特征门则输出一个向量，该向量的每个元素都与特征嵌入的每一位相关，从而可以提供更细粒度的重要性信息。
>
> 此外，文章还提到了参数共享的概念，即“字段共享”和“字段私有”。在“字段共享”中，所有的特征门都使用相同的参数，这有助于减少模型的总参数数量。相反，在“字段私有”中，每个特征门都有自己的参数，这允许模型为每个特征学习特定的重要性权重。这两种方法的性能将在后续章节中进行比较。

#### 3.2 Hidden Gate

The deep part of many DNN ranking systems usually consists of several full-connected layers, which implicitly captures high-order features interactions. As shown in Figure 1, the input of deep network is the flatten of embedding layer.Let $a^{(0)}=\left[g e_1, \cdots, g e_i, \cdots, g e_f\right]$ denotes the outputs of embedding layer, where $ge_i \in R^k$ represents the $i$−th feature embedding. Then, $a^{(0)}$ is fed into multi-layer perceptron network, and the feed forward process is:
$$
a^{(l)}=\sigma\left(W^{(l)} a^{(l-1)}+b^{(l)}\right)
$$
where $l$ is the depth and $\sigma$ is the activation function. $W(l)$, $b(l)$ ,$a(l)$ are the model weight, bias and output of the $l$-th layer.

Similar to the bit-wise feature embedding gate, we proposed the hidden gate which can be applied to the hidden layer. As depicted as Figure 3, we use this gate as follows:
$$
g^{(l)}=a^{(l)} \odot \sigma_g\left(W_g^{(l)} a^{(l)}\right)
$$
where $⊙$ denotes the element-wise product, $\sigma_g$ is the gate activation function, $W^{(l)}_g$ is the $l$​-th layer parameter of hidden gate.Likewise, we can stack multiple hidden gate layers like the classic DNN models.

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Figure3.png)

> **翻译**：
>
> 许多深度神经网络（DNN）排序系统的深层部分通常由几个全连接层组成，这些层可以隐式地捕获高阶特征的交互。如图1所示，深度网络的输入是嵌入层的扁平化结果。设$a^{(0)}=\left[g e_1, \cdots, g e_i, \cdots, g e_f\right]$为嵌入层的输出，其中$ge_i \in R^k$表示第$i$个特征的嵌入。然后，$a^{(0)}$被送入多层感知器网络，前馈过程如下：
>
> $$
> a^{(l)}=\sigma\left(W^{(l)} a^{(l-1)}+b^{(l)}\right)
> $$
>
> 其中 $l$ 是深度，$\sigma$ 是激活函数。$W(l)$，$b(l)$，$a(l)$分别是第$l$层的模型权重、偏置和输出。
>
> 与位级特征嵌入门类似，我们提出了可应用于隐藏层的隐藏门。如图3所示，我们使用这个门的方式如下：
>
> $$
> g^{(l)}=a^{(l)} \odot \sigma_g\left(W_g^{(l)} a^{(l)}\right)
> $$
>
> 其中$⊙$表示元素乘积，$\sigma_g$是门的激活函数，$W^{(l)}_g$是隐藏门的第 $l$ 层参数。同样，我们可以像经典的DNN模型一样堆叠多个隐藏门层。
>
> **理解**：
>
> 这段论文描述了一个深度神经网络（DNN）排序系统的一部分，特别是其深层结构。这部分主要由全连接层组成，用于捕获高阶特征的交互。输入到这些深层结构的数据首先经过嵌入层处理，并输出一个扁平化的嵌入向量。这个嵌入向量随后被送入一个多层感知器网络。
>
> 在多层感知器网络中，数据通过一系列的前馈过程进行传递，每个过程都涉及到一个权重矩阵、一个偏置向量和一个激活函数。这个过程是神经网络中常见的操作，用于学习和模拟输入数据中的复杂关系。
>
> 此外，论文还介绍了一种名为“隐藏门”的机制。这个机制与先前介绍的位级特征嵌入门类似，但它是应用于隐藏层的。隐藏门通过使用元素乘积和一个特定的激活函数来修改隐藏层的输出。这种机制可能有助于网络更好地学习和适应数据中的复杂模式。通过堆叠多个隐藏门层，网络可以构建更复杂的表示和学习能力。

#### 3.3 Output Layer

To summarize, we give the overall formulation of our proposed model’output as:
$$
\hat{y}=\sigma\left(W^{|L|} g^{|L|}+b^{|L|}\right)
$$
where $\hat{y} \in (0,1)$ is the predicted value of CTR, $\sigma$ is the sigmoid function, $b^{|L|}$ is the bias and $|L|$ is the depth of DNN. The learning process aims to minimize the following objective function (cross entropy):
$$
\text { loss }=-\frac{1}{N} \sum_{i=1}^N\left(y_i \log \left(\hat{y}_i\right)+\left(1-y_i\right) * \log \left(1-\hat{y}_i\right)\right)
$$
where $y_i$ is the ground truth of $i$-th instance, $\hat{y}_iˆ$   is the predicted CTR, and $N$ is the total size of samples.

> **翻译**：
>
> 综上所述，我们给出所提出模型的总体输出公式为：
>
> $$
> \hat{y}=\sigma\left(W^{|L|} g^{|L|}+b^{|L|}\right)
> $$
>
> 其中，$\hat{y} \in (0,1)$ 是预测的点击率（CTR）值，$\sigma$ 是sigmoid函数，$b^{|L|}$ 是偏置项，而 $|L|$ 是深度神经网络（DNN）的深度。学习过程旨在最小化以下目标函数（交叉熵）：
>
> $$
> \text { loss }=-\frac{1}{N} \sum_{i=1}^N\left(y_i \log \left(\hat{y}_i\right)+\left(1-y_i\right) * \log \left(1-\hat{y}_i\right)\right)
> $$
>
> 其中，$y_i$ 是第$i$个实例的真实值，$\hat{y}_i$ 是预测的CTR，而 $N$ 是样本的总数。
>

## 4 EXPERIMENTS

In this section, we conduct extensive experiments to answer the following research questions:

- (RQ1) Can the feature embedding gate enhance the ability of the baseline models?
- (RQ2) Can the hidden gate enhance the ability of the baseline models?
- (RQ3) Can we combine the two gates in one model to achieve further improvements?
- (RQ4) How do the settings of networks influence the performance of our model?

We will answer these questions after presenting some fundamental experimental settings.

> 在本节中，我们将进行广泛的实验来回答以下研究问题：
>
> - (RQ1) 特征嵌入门能否增强基线模型的能力？
> - (RQ2) 隐藏门能否增强基线模型的能力？
> - (RQ3) 我们能否在一个模型中结合这两个门以获得进一步的改进？
> - (RQ4) 网络设置如何影响我们模型的性能？
>
> 在介绍了一些基本的实验设置之后，我们将回答这些问题。
>

#### 4.1 Experimental Testbeds and Setup

*4.1.1 Data Sets.* 1) Criteo. The Criteo1 dataset is widely used in many CTR model evaluation. It contains click logs with 45 millions data instances. There are 26 anonymous categorical fields and 13 continuous feature fields in Criteo dataset. We split the dataset ran- domly into two parts: 90% is for training, while the rest is for testing. 2) ICME. The ICME2 dataset consists of several days of short video click datas. It contains click logs with 19 millions data instances in track2. For each click data, we choose 5 fields(user_id, user_city, item_id,author_id,item_city) to predict the like probability of short video. We split it randomly into two parts: 70% is for training, while the rest is for testing. 3) SafeDriver. The SafeDriver3 dataset is used to predict the probability that an auto insurance policy holder files a claim. There are 57 anonymous fields in SafeDriver dataset and these features are divided into similar groups:binary features, categorical features, continuous features and ordinal features. It contains 595K data instances. We split the dataset randomly into two parts: 90% is for training, while the rest is for testing.

*4.1.2 Evaluation Metrics*. In our experiment, we adopt AUC(Area Under ROC) as metric. AUC is a widely used metric in evaluating classification problems. Besides, some work validates AUC as a good measurement in CTR prediction[8]. AUC is insensitive to the classification threshold and the positive ratio. The upper bound of AUC is 1, and the larger the better.

*4.1.3 Baseline Methods*. To verify the effect of the gate layer added in various mainstream models, we choose some widely used CTR models as our baseline models including FM[21, 22], DNN, DeepFM[9], and XDeepFM[16].

Main goal of this work is not intent to propose a new model instead of enhancing these baseline models via gating mechanism that we proposed. Note that an improvement of 1‰ in AUC is usually regarded as significant for the CTR prediction because it will bring a large increase in a company’s revenue if the company has a very large user base.

*4.1.4 Implementation Details*. We implement all the models with Tensorflow4 in our experiments. For the embedding layer, the di- mension of embedding layer is set to 10. For the optimization method, we use the Adam[13] with a mini-batch size of 1000, and the learning rate is set to 0.0001. For all deep models, the depth of layers is set to 3, all activation functions are RELU, the number of neurons per layer is 400, and the dropout rate is set to 0.5. The default activation function of feature embedding gate is Sigmoid and activation function of hidden gate is Tanh. We conduct our experiments with 2 Tesla K40 GPUs.

> #### 
>
> *4.1.1 数据集* 
>
> 1) Criteo。Criteo1数据集在许多CTR模型评估中广泛使用。它包含4500万个数据实例的点击日志。Criteo数据集中有26个匿名分类字段和13个连续特征字段。我们将数据集随机分成两部分：90%用于训练，其余用于测试。
> 2) ICME。ICME2数据集包含几天的短视频点击数据。它包含track2中1900万个数据实例的点击日志。对于每个点击数据，我们选择5个字段（user_id、user_city、item_id、author_id、item_city）来预测短视频的点赞概率。我们随机将其分为两部分：70%用于训练，其余用于测试。
> 3) SafeDriver。SafeDriver3数据集用于预测汽车保险投保人提出索赔的概率。SafeDriver数据集中有57个匿名字段，这些特征被分为类似的组：二元特征、分类特征、连续特征和有序特征。它包含595K个数据实例。我们将数据集随机分成两部分：90%用于训练，其余用于测试。
>
> *4.1.2 评估指标*
> 在我们的实验中，我们采用AUC（Area Under ROC）作为指标。AUC是评估分类问题的广泛使用指标。此外，一些工作验证了AUC是CTR预测中的一个良好衡量标准[8]。AUC对分类阈值和正比率不敏感。AUC的上限是1，越大越好。
>
> *4.1.3 基线方法*
> 为了验证在各种主流模型中添加门层的效果，我们选择了一些广泛使用的CTR模型作为我们的基线模型，包括FM[21, 22]、DNN、DeepFM[9]和XDeepFM[16]。
>
> 这项工作的主要目标不是提出一个新模型，而是通过我们提出的门控机制来增强这些基线模型。请注意，对于CTR预测，AUC提高1‰通常被认为是非常重要的，因为如果公司拥有非常大的用户群，这将为公司带来巨大的收入增长。
>
> *4.1.4 实施细节*
> 在我们的实验中，我们使用Tensorflow4实现了所有模型。对于嵌入层，嵌入层的维度设置为10。对于优化方法，我们使用Adam[13]，小批量大小为1000，学习率设置为0.0001。对于所有深度模型，层数设置为3，所有激活函数都是RELU，每层神经元数量为400，dropout率设置为0.5。特征嵌入门的默认激活函数是Sigmoid，隐藏门的激活函数是Tanh。我们使用2个Tesla K40 GPU进行实验。

#### 4.2 Performance of Feature Embedding Gate(RQ1)

In this subsection, we show the performance gains of chosen base- line models after inserting feature embedding gate into a typical embedding layer. The experiments are conducted on Criteo,ICME and SafeDriver datasets and results are shown in Table 1.

> 在本小节中，我们展示了在将特征嵌入门插入到典型的嵌入层之后，所选基线模型的性能提升。实验在Criteo、ICME和SafeDriver数据集上进行，结果如表1所示。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table1.png)

Inserting the feature embedding gate into these baseline models, we find our proposed embedding gate mechanisms can consistently boost the baseline model’s performance on these three datasets as shown in Table 1. These results indicate that carefully selecting salient latent information from the feature-level is useful to enhance the model ability and make the baseline models achieve better performance. Among all the baseline models, FM with the feature embedding gate gets a significant improvement which outperforms the classic FM model by almost 2% on ICME dataset. We assume that FM is a shallow model that has only a set of latent vectors to learn, there’s no other component in FM to explicitly or implicitly adjust the feature in FM, so the gate mechanism is a good way to adjust the feature weight. Instead of FM, there are many deep models such as DeepFM and XDeepFM, our models with feature embedding gate can enhance these models’ ability and make further improvements.

Moreover, we design some further research about feature embedding gate. First, we conduct some experiments to compare param ter sharing mechanism of gate(‘field sharing’ and ‘field private’) in Table 2.Although the ‘field sharing’ can reduce the number of learned pa- rameters, the performance also decreases. These results indicate that the performance of different parameter sharing mechanisms of gate depend on specific task. On the whole, it is a better choice to choose the ‘field private’ in our experiments.

> 将特征嵌入门插入到这些基线模型中，我们发现我们提出的嵌入门机制可以在这三个数据集上一致地提升基线模型的性能，如表1所示。这些结果表明，从特征级别仔细选择显著的潜在信息对于增强模型能力和使基线模型获得更好的性能是有用的。在所有基线模型中，带有特征嵌入门的FM模型得到了显著提升，在ICME数据集上比经典的FM模型提高了近2%。我们认为，FM是一个浅层模型，只有一组潜在向量可以学习，FM中没有其他组件可以显式或隐式地调整特征，因此门机制是调整特征权重的好方法。除了FM之外，还有许多深度模型，如DeepFM和XDeepFM，我们的带特征嵌入门的模型可以提升这些模型的能力并做出进一步的改进。
>
> 此外，我们对特征嵌入门进行了一些进一步的研究。首先，我们进行了一些实验，以比较门的不同参数共享机制（'field sharing'和'field private'），如表2所示。尽管'field sharing'可以减少学习参数的数量，但性能也会下降。这些结果表明，门的不同参数共享机制的性能取决于特定的任务。总的来说，在我们的实验中，选择'field private'是一个更好的选择。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table2.png)

From the Table 2, we can find that the performance of ‘field pri- vate’ gate is much better than the ‘field sharing’ gate for many base models on ICME dataset while it is not significant on Criteo dataset.

Although the ‘field sharing’ can reduce the number of learned pa- rameters, the performance also decreases. These results indicate that the performance of different parameter sharing mechanisms of gate depend on specific task. On the whole, it is a better choice to choose the ‘field private’ in our experiments.

Second, we conduct some experiments to explore the vector-wise and bit-wise feature embedding gate. The results in Table 3 show that bit-wise is a little better than vector-wise on Criteo dataset, while we cannot draw an obvious conclusion on the ICME data. The reason behind this needs further exploration.

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table3.png)

> 从表2中，我们可以发现，对于ICME数据集上的许多基础模型来说，‘field private’门的性能要优于‘field sharing’门，但在Criteo数据集上则不太显著。
>
> 尽管‘field sharing’可以减少学习参数的数量，但性能也会下降。这些结果表明，不同的参数共享机制的门性能取决于特定任务。总的来说，在我们的实验中，选择‘field private’是更好的选择。
>
> 其次，我们进行了一些实验来探索向量级和位级的特征嵌入门。表3的结果表明，在Criteo数据集上，位级略微优于向量级，但在ICME数据上，我们无法得出明显的结论。这背后的原因需要进一步探索。

#### 4.3 Performance of Hidden Gate(RQ2)

In this subsection, the overall performance gains of chosen baseline models after inserting hidden gate into a typical MLP layer will be reported on these three test sets in Table 4.

Replacing the traditional MLP with the hidden gate layer, our proposed hidden gate mechanisms consistently enhance these base- line models and achieve performance improvements on the ICME, Criteo and SafeDriver dataset as shown in Table 4. The experimen- tal results indicate that the hidden gate helps the model to implicitly capture the high-order interaction more effectively.

Although applying hidden gate to MLP layers is simple, it is an effective way to improve the performance of baseline models. Therefore, we conduct experiments to compare hidden gate DNN with some complex base models in the Table 5.

From the Table 5, the standard DNN by inserting hidden gate outperforms some canonical deep learning models such as DeepFM, XDeepFM. It is a simple way to enhance the standard DNN to gain improvement, which makes the DNN model much more practicable in industrial recommendation systems.

> 在本小节中，我们将在表4中报告在将隐藏门插入到典型的MLP层后，所选基线模型在这三个测试集上的整体性能提升。
>
> 用隐藏门层替换传统的MLP，我们提出的隐藏门机制能够持续提升这些基线模型，并在ICME、Criteo和SafeDriver数据集上实现性能提升，如表4所示。实验结果表明，隐藏门有助于模型更有效地隐式捕获高阶交互。
>
> 尽管在MLP层中应用隐藏门很简单，但它是提高基线模型性能的有效方法。因此，我们在表5中进行了实验，以比较隐藏门DNN和一些复杂的基线模型。
>
> 从表5中可以看出，通过插入隐藏门的标准DNN优于一些典型的深度学习模型，如DeepFM、XDeepFM。这是一种提升标准DNN性能的简单方法，使得DNN模型在工业推荐系统中更加实用。

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table4.png)

![Table5](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table5.png)

#### 4.4 Performance of model Combining FE-Gate and Hidden Gate(RQ3)

As mentioned previously, we find the feature embedding gate and hidden gate can enhance the model ability and gain good perfor- mance, respectively. Can we combine the feature embedding gate and hidden gate in one model to achieve further performance? We conduct some experiments to answer this research question on Criteo and ICME datasets.

It can be seen from Table 6 that combining feature embedding gate and hidden gate in one model can not gain further perfor- mance improvements. Specifically, there is not much performance improvements on Criteo and some performance decrease on ICME. The feature embedding gate can influence the implicit and explicit feature interaction while the hidden gate can influence the implicit feature interaction, we assume that the implicit feature interac- tions have been done twice and the implicit feature representations are damaged. The real reason behind this need to conduct further experiments to justify this assumption.

> 

![Table6](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table6.png)

#### 4.5 Hyper-parameter Study(RQ4)

We conduct some experiments to study the influence of hyper- parameter in our proposed gate mechanisms. We test different settings in our proposed GateNet on the SafeDriver dataset and we treat DeepFM, DeepFMe and DeepFMh as the baseline models.

So we divide the hyper-parameters into the following three parts:

- Gate activation function. Both embedding and hidden gate include the gate activation functions.
- Embedding size. We change the embedding size from 10 to 50, and compare the performance of baseline model with embedding gate model.
- Hidden layers. We change the number of layers from 2 to 6, and observe the performance of baseline model and hidden gate model.

*4.5.1 Activation function in Gate*. The test results on SafeDriver dataset with different activation functions in the feature embedding gate and hidden gate are presented in Table 7. We observe that the best activation function in feature embedding gate is linear while the best activation function is Tanh in hidden gate.

*4.5.2 Embedding Size in Feature Embedding Gate*. We change the embedding size from 10 to 50 in feature embedding gate and summa- rize the range of performances in Table 8. From the results, we find that embedding size has little influence on the GateNet. Specifically, the standard DeepFM has a good performance with the embedding size 20, while the embedding size of DeepFMe is 10. Therefore, these results show that DeepFMe requires less parameter than DeepFM to train a good model.

*4.5.3 Number of Layers in Hidden Gate*. In deep part, we can change the number of neurons per layer, depths of DNN, activation functions and dropout rates. For brevity, we just study the impact of different depths in DNN part. We change the number of layers from 2 to 6 in hidden gate and conclude the performance in Table 9.

Increasing the number of layers, the performance of DeepFM increases, while DeepFMh decreases. These results indicate that our DeepFM can learn much better than DeepFM with less parameters on SafeDriver dataset.

> 我们进行了一些实验来研究我们提出的门机制中超参数的影响。我们在SafeDriver数据集上测试了我们提出的GateNet的不同设置，并将DeepFM、DeepFMe和DeepFMh作为基线模型。
>
> 因此，我们将超参数分为以下三部分：
>
> - 门激活函数。嵌入门和隐藏门都包括门激活函数。
>
> - 嵌入大小。我们将嵌入大小从10改为50，并比较基线模型和嵌入门模型的性能。
>
> - 隐藏层。我们将层数从2层改为6层，并观察基线模型和隐藏门模型的性能。
>
> *4.5.1 门中的激活函数*。表7展示了在SafeDriver数据集上使用特征嵌入门和隐藏门中的不同激活函数的测试结果。我们观察到，特征嵌入门中的最佳激活函数是线性的，而隐藏门中的最佳激活函数是双曲正切函数（Tanh）。
>
> *4.5.2 特征嵌入门中的嵌入大小*。我们在特征嵌入门中将嵌入大小从10改为50，并在表8中总结了性能范围。从结果中，我们发现嵌入大小对GateNet的影响很小。具体来说，标准DeepFM在嵌入大小为 20 时表现良好，而DeepFMe的嵌入大小为10。因此，这些结果表明，与DeepFM相比，DeepFMe需要更少的参数来训练一个好的模型。
>
> *4.5.3 隐藏门中的层数*。在深度部分，我们可以改变每层的神经元数量、DNN的深度、激活函数和dropout率。为了简洁起见，我们只研究DNN部分不同深度的影响。我们在隐藏门中将层数从2层改为6层，并在表9中总结了性能。
>
> 随着层数的增加，DeepFM的性能提高，而DeepFMh的性能下降。这些结果表明，在SafeDriver数据集上，我们的DeepFM可以比DeepFM用更少的参数学习得更好。

![Table8](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table8.png)

![Table7](/Users/anmingyu/Github/Gor-rok/Papers/multitask/GATE/Table7.png)

## 5 CONCLUSIONS

Recently, many neural network based CTR models have been pro- posed and some recent research results found that gating mech- anisms can improve the trainability in training non-convex deep neural networks. Inspired by these observations, we proposed a novel model named GateNet which introduces either the feature embedding gate or the hidden gate to the embedding layer or hid- den layers of DNN CTR models,respectively. Extensive experiments conducted on three real-world datasets demonstrate its effective- ness to boost the performance of various state-of-the-art models such as FM, DeepFM and xDeepFM on three real-world datasets.

> 最近，许多基于神经网络的点击率（CTR）模型相继被提出，而一些最新的研究结果发现，门控机制可以提高非凸深度神经网络训练的可训练性。受这些观察的启发，我们提出了一种名为GateNet的新型模型，该模型将特征嵌入门或隐藏门分别引入到深度神经网络（DNN）CTR模型的嵌入层或隐藏层中。在三个真实数据集上进行的广泛实验表明，该模型可以有效地提升各种最先进的模型（如FM、DeepFM和xDeepFM）在真实数据集上的性能。
>