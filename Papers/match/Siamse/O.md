# Siamese Neural Networks for One-shot Image Recognition

## Abstract

The process of learning good features for machine learning applications can be very computationally expensive and may prove difficult in cases where little data is available. A prototypical example of this is the one-shot learning setting, in which we must correctly make predictions given only a single example of each new class. In this paper, we explore a method for learning siamese neural networks which employ a unique structure to naturally rank similarity between inputs. Once a network has been tuned, we can then capitalize on powerful discriminative features to generalize the predictive power of the network not just to new data, but to entirely new classes from unknown distributions. Using a convolutional architecture, we are able to achieve strong results which exceed those of other deep learning models with near state-of-the-art performance on one-shot classification tasks.

> 为机器学习应用学习好的特征的过程可能计算成本非常高，且在数据量很少的情况下可能很困难。这方面的一个典型例子就是一次性学习设置，在这种设置中，我们必须只根据每个新类别的一个例子来做出正确的预测。在本文中，我们探索了一种学习连体神经网络的方法，该方法采用独特的结构来对输入之间的相似性进行自然排序。一旦网络被调整，我们就可以利用强大的判别特征，将网络的预测能力不仅推广到新的数据，而且推广到来自未知分布的全新类别。通过使用卷积架构，我们能够取得强大的结果，在一分类任务上接近最先进的性能，超过了其他深度学习模型。

Humans exhibit a strong ability to acquire and recognize new patterns. In particular, we observe that when presented with stimuli, people seem to be able to understand new concepts quickly and then recognize variations on these concepts in future percepts (Lake et al., 2011). Machine learning has been successfully used to achieve state-ofthe-art performance in a variety of applications such as web search, spam detection, caption generation, and speech and image recognition. However, these algorithms often break down when forced to make predictions about data for which little supervised information is available. We desire to generalize to these unfamiliar categories without necessitating extensive retraining which may be either expensive or impossible due to limited data or in an online prediction setting, such as web retrieval.

> 人类表现出强大的获取和识别新模式的能力。尤其是，我们观察到，当受到刺激时，人们似乎能够迅速理解新概念，然后在未来的感知中识别出这些概念的变化（Lake等人，2011年）。机器学习已成功应用于网页搜索、垃圾邮件检测、字幕生成以及语音和图像识别等多种应用中，并取得了最先进的性能。然而，当被迫对监督信息很少的数据进行预测时，这些算法往往会崩溃。我们希望在不需要大量重新训练的情况下，将这些算法推广到这些不熟悉的类别，因为在数据有限或在线预测设置（如网络检索）中，重新训练可能成本高昂或无法实现。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/match/Siamse/Figure1.png)

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/match/Siamse/Figure2.png)

One particularly interesting task is classification under the restriction that we may only observe a single example of each possible class before making a prediction about a test instance. This is called one-shot learning and it is the primary focus of our model presented in this work (Fei-Fei et al., 2006; Lake et al., 2011). This should be distinguished from zero-shot learning, in which the model cannot look at any examples from the target classes (Palatucci et al., 2009).

One-shot learning can be directly addressed by developing domain-specific features or inference procedures which possess highly discriminative properties for the target task. As a result, systems which incorporate these methods tend to excel at similar instances but fail to offer robust solutions that may be applied to other types of problems. In this paper, we present a novel approach which limits assumptions on the structure of the inputs while automatically acquiring features which enable the model to generalize successfully from few examples. We build upon the deep learning framework, which uses many layers of non-linearities to capture invariances to transformation in the input space, usually by leveraging a model with many parameters and then using a large amount of data to prevent overfitting (Bengio, 2009; Hinton et al., 2006). These features are very powerful because we are able to learn them without imposing strong priors, although the cost of the learning algorithm itself may be considerable.

> 一个特别有趣的任务是在对测试实例进行预测之前，我们可能只观察到每个可能类别的单个example 的限制下进行分类。这被称为一次性学习，也是本工作中提出的模型的主要关注点（Fei-Fei等，2006；Lake等，2011）。这应与零样本学习区分开来，在零样本学习中，模型不能查看目标类别的任何示例（Palatucci等，2009）。
>
> 可以通过开发具有高度判别性的针对目标任务的域特定特征或推理程序来直接解决一次性学习问题。因此，采用这些方法的系统往往擅长处理类似实例，但无法提供可应用于其他类型问题的稳健解决方案。在本文中，我们提出了一种新方法，该方法在自动获取特征的同时，限制了对输入结构的假设，从而使模型能够从少数示例中成功推广。我们的方法是建立在深度学习框架之上的，该框架使用多层非线性来捕获输入空间中变换的不变性，通常是通过利用具有许多参数的模型，然后使用大量数据来防止过拟合（Bengio，2009；Hinton等，2006）。这些特征非常强大，因为我们能够在不施加强烈先验的情况下学习它们，尽管学习算法本身的成本可能相当高。



## 1. Approach

In general, we learn image representations via a supervised metric-based approach with siamese neural networks, then reuse that network’s features for one-shot learning without any retraining.

In our experiments, we restrict our attention to character recognition, although the basic approach can be replicated for almost any modality (Figure 2). For this domain, we employ large siamese convolutional neural networks which a) are capable of learning generic image features useful for making predictions about unknown class distributions even when very few examples from these new distributions are available; b) are easily trained using standard optimization techniques on pairs sampled from the source data; and c) provide a competitive approach that does not rely upon domain-specific knowledge by instead exploiting deep learning techniques.

To develop a model for one-shot image classification, we aim to first learn a neural network that can discriminate between the class-identity of image pairs, which is the standard verification task for image recognition. We hypothesize that networks which do well at at verification should generalize to one-shot classification. The verification model learns to identify input pairs according to the probability that they belong to the same class or different classes. This model can then be used to evaluate new images, exactly one per novel class, in a pairwise manner against the test image. The pairing with the highest score according to the verification network is then awarded the highest probability for the one-shot task. If the features learned by the verification model are sufficient to confirm or deny the identity of characters from one set of alphabets, then they ought to be sufficient for other alphabets, provided that the model has been exposed to a variety of alphabets to encourage variance amongst the learned features.

> 总的来说，我们通过基于监督度量的方法，使用孪生神经网络来学习图像表示，然后重用该网络的特征进行一次性学习，而无需任何重新训练。
>
> 在我们的实验中，我们将注意力集中在字符识别上，尽管基本方法几乎可以复制到任何模式（见图2）。对于此领域，我们使用了大型孪生卷积神经网络，该网络
>
> a）能够学习通用图像特征，这些特征对于根据这些新分布的极少示例对未知类别分布进行预测很有用；
>
> b）使用从源数据中采样的对，通过标准优化技术容易进行训练；
>
> c）利用深度学习技术提供了一种不依赖领域特定知识的竞争方法。
>
> 为了开发一次性图像分类模型，我们的目标是首先学习一个能够区分图像对类别身份的神经网络，这是图像识别的标准验证任务。我们假设在验证方面表现良好的网络应该能够推广到一次性分类。验证模型学习根据输入对属于同一类别或不同类别的概率来识别它们。然后，可以使用此模型以成对的方式对每个新类别的新图像进行评估，与测试图像进行对比。然后，根据验证网络的最高得分，为一次性任务分配最高概率的配对。如果验证模型学习的特征足以确认或否认一组字母中字符的身份，那么只要模型已经接触过各种字母以鼓励学习特征之间的差异，它们就应该足以用于其他字母。

## 2. Related Work

Overall, research into one-shot learning algorithms is fairly immature and has received limited attention by the machine learning community. There are nevertheless a few key lines of work which precede this paper.

The seminal work towards one-shot learning dates back to the early 2000’s with work by Li Fei-Fei et al. The authors developed a variational Bayesian framework for oneshot image classification using the premise that previously learned classes can be leveraged to help forecast future ones when very few examples are available from a given class (Fe-Fei et al., 2003; Fei-Fei et al., 2006). More recently, Lake et al. approached the problem of one-shot learning from the point of view of cognitive science, addressing one-shot learning for character recognition with a method called Hierarchical Bayesian Program Learning (HBPL) (2013). In a series of several papers, the authors modeled the process of drawing characters generatively to decompose the image into small pieces (Lake et al., 2011; 2012). The goal of HBPL is to determine a structural explanation for the observed pixels. However, inference under HBPL is difficult since the joint parameter space is very large, leading to an intractable integration problem.

Some researchers have considered other modalities or transfer learning approaches. Lake et al. have some very recent work which uses a generative Hierarchical Hidden Markov model for speech primitives combined with a Bayesian inference procedure to recognize new words by unknown speakers (2014). Maas and Kemp have some of the only published work using Bayesian networks to predict attributes for Ellis Island passenger data (2009). Wu and Dennis address one-shot learning in the context of path planning algorithms for robotic actuation (2012). Lim focuses on how to “borrow” examples from other classes in the training set by adapting a measure of how much each category should be weighted by each training exemplar in the loss function (2012). This idea can be useful for data sets where very few examples exist for some classes, providing a flexible and continuous means of incorporating inter-class information into the model.

> 总的来说，一次性学习算法的研究还相当不成熟，并且机器学习领域对此的关注也有限。然而，在这篇论文之前，确实有一些关键的研究工作。
>
> 一次性学习的开创性工作可以追溯到21世纪初，由李飞飞等人完成。他们开发了一个变分贝叶斯框架，用于一次性图像分类，前提是当给定类别的可用示例非常少时，可以利用先前学习的类别来帮助预测未来的类别（Fe-Fei等，2003；Fei-Fei等，2006）。最近，Lake等人从认知科学的角度研究了一次性学习问题，他们使用一种称为分层贝叶斯程序学习（HBPL）的方法来解决字符识别的一次性学习问题（2013）。在一系列论文中，作者通过生成性地绘制字符来模拟将图像分解成小块的过程（Lake等，2011；2012）。HBPL的目标是确定观察到的像素的结构解释。然而，HBPL下的推理是困难的，因为联合参数空间非常大，导致了一个难以处理的积分问题。
>
> 一些研究人员考虑了其他模式或迁移学习方法。Lake等人最近的一些工作使用了一个生成性的分层隐马尔可夫模型来处理语音基元，并结合贝叶斯推理程序来识别未知说话者的新单词（2014）。Maas和Kemp发表了唯一使用贝叶斯网络预测埃利斯岛乘客数据的属性的工作（2009）。Wu和Dennis在机器人驱动的路径规划算法的背景下解决了一次性学习问题（2012）。Lim专注于如何通过调整损失函数中每个训练样本应对每个类别的加权程度来“借用”训练集中的其他类别的示例（2012）。这个想法对于一些类别中示例非常少的数据集非常有用，它提供了一种灵活且连续的方式将类间信息融入模型中。



## 3. Deep Siamese Networks for Image Verification

Siamese nets were first introduced in the early 1990s by Bromley and LeCun to solve signature verification as an image matching problem (Bromley et al., 1993). A siamese neural network consists of twin networks which accept distinct inputs but are joined by an energy function at the top. This function computes some metric between the highestlevel feature representation on each side (Figure 3). The parameters between the twin networks are tied. Weight tying guarantees that two extremely similar images could not possibly be mapped by their respective networks to very different locations in feature space because each network computes the same function. Also, the network is symmetric, so that whenever we present two distinct images to the twin networks, the top conjoining layer will compute the same metric as if we were to we present the same two images but to the opposite twins.

In LeCun et al., the authors used a contrastive energy function which contained dual terms to decrease the energy of like pairs and increase the energy of unlike pairs (2005). However, in this paper we use the weighted L1 distance between the twin feature vectors h1 and h2 combined with a sigmoid activation, which maps onto the interval [0, 1]. Thus a cross-entropy objective is a natural choice for training the network. Note that in LeCun et al., they directly learned the similarity metric, which was implictly defined by the energy loss, whereas we fix the metric as specified above, following the approach in Facebook’s DeepFace paper (Taigman et al., 2014).

Our best-performing models use multiple convolutional layers before the fully-connected layers and top-level energy function. Convolutional neural networks have achieved exceptional results in many large-scale computer vision applications, particularly in image recognition tasks (Bengio, 2009; Krizhevsky et al., 2012; Simonyan & Zisserman, 2014; Srivastava, 2013).

Several factors make convolutional networks especially appealing. Local connectivity can greatly reduce the number of parameters in the model, which inherently provides some form of built-in regularization, although convolutional layers are computationally more expensive than standard nonlinearities. Also, the convolution operation used in these networks has a direct filtering interpretation, where each feature map is convolved against input features to identify patterns as groupings of pixels. Thus, the outputs of each convolutional layer correspond to important spatial features in the original input space and offer some robustness to simple transforms. Finally, very fast CUDA libraries are now available in order to build large convolutional networks without an unacceptable amount of training time (Mnih, 2009; Krizhevsky et al., 2012; Simonyan & Zisserman, 2014).

We now detail both the structure of the siamese nets and the specifics of the learning algorithm used in our experiments.

> 孪生网络最早由Bromley和LeCun在20世纪90年代初引入，用于解决签名验证的图像匹配问题（Bromley等，1993）。孪生神经网络由两个接受不同输入但顶部连接一个能量函数的网络组成。这个函数计算两边最高级特征表示之间的某种度量（图3）。两个网络之间的参数是绑定的。权重绑定保证了两个极其相似的图像不可能被它们各自的网络映射到特征空间中非常不同的位置，因为每个网络都计算相同的函数。此外，网络是对称的，因此每当我们向两个网络呈现两个不同的图像时，顶部的连接层将计算与我们向相反的两个网络呈现相同图像时相同的度量。
>
> 在LeCun等人的研究中，作者使用了一种包含双重项的对比能量函数，以降低相似对的能量并增加不相似对的能量（2005）。然而，在本文中，我们使用了孪生特征向量 h1和h2之间的加权L1距离，并结合了一个sigmoid激活函数，该函数映射到区间[0, 1]。因此，交叉熵目标是训练网络的自然选择。请注意，在LeCun等人的研究中，他们直接学习了由能量损失隐式定义的相似性度量，而我们按照Facebook的DeepFace论文中的方法将度量固定为上述规定（Taigman等，2014）。
>
> 我们表现最好的模型在全连接层和顶级能量函数之前使用了多个卷积层。卷积神经网络在许多大规模计算机视觉应用中取得了卓越的成果，特别是在图像识别任务中（Bengio，2009；Krizhevsky等，2012；Simonyan&Zisserman，2014；Srivastava，2013）。
>
> 有几个因素使卷积网络特别吸引人。局部连接可以大大减少模型中的参数数量，这本身就提供了一定形式的内置正则化，尽管卷积层在计算上比标准非线性更昂贵。此外，这些网络中使用的卷积操作具有直接的滤波解释，其中每个特征图都与输入特征进行卷积，以将像素分组为图案。因此，每个卷积层的输出对应于原始输入空间中的重要空间特征，并对简单的变换提供了一定的鲁棒性。最后，现在可以使用非常快速的CUDA库来构建大型卷积网络，而无需花费太多训练时间（Mnih，2009；Krizhevsky等，2012；Simonyan&Zisserman，2014）。
>
> 现在，我们将详细介绍孪生网络的结构以及我们在实验中使用的学习算法的具体内容。

#### 3.1. Model

Our standard model is a siamese convolutional neural network with $L$ layers each with $N_l$ units, where $h_{1,l}$ represents the hidden vector in layer $l$ for the first twin, and $h_{2,l}$ denotes the same for the second twin. We use exclusively rectified linear (ReLU) units in the first $L − 2$ layers and sigmoidal units in the remaining layers.

The model consists of a sequence of convolutional layers, each of which uses a single channel with filters of varying size and a fixed stride of $1$. The number of convolutional filters is specified as a multiple of $16$ to optimize performance. The network applies a ReLU activation function to the output feature maps, optionally followed by maxpooling with a filter size and stride of $2$. Thus the kth filter map in each layer takes the following form:
$$
\begin{aligned}
& a_{1, m}^{(k)}=\operatorname{max-pool}\left(\max \left(0, \mathbf{W}_{l-1, l}^{(k)} \star \mathbf{h}_{1,(l-1)}+\mathbf{b}_l\right), 2\right) \\
& a_{2, m}^{(k)}=\operatorname{max-pool}\left(\max \left(0, \mathbf{W}_{l-1, l}^{(k)} \star \mathbf{h}_{2,(l-1)}+\mathbf{b}_l\right), 2\right)
\end{aligned}
$$
where $W_{l−1,l}$ is the 3-dimensional tensor representing the feature maps for layer $l$ and we have taken $\star$ to be the valid convolutional operation corresponding to returning only those output units which were the result of complete overlap between each convolutional filter and the input feature maps.

The units in the final convolutional layer are flattened into a single vector. This convolutional layer is followed by a fully-connected layer, and then one more layer computing the induced distance metric between each siamese twin, which is given to a single sigmoidal output unit. More precisely, the prediction vector is given as $\mathcal{p} = \sigma\left(\sum_j \alpha_j\left|\mathbf{h}_{1, L-1}^{(j)}-\mathbf{h}_{2, L-1}^{(j)}\right|\right)$, where $σ$ is the sigmoidal activation function. This final layer induces a metric on the learned feature space of the ($L − 1$)th hidden layer and scores the similarity between the two feature vectors. The $α_j$ are additional parameters that are learned by the model during training, weighting the importance of the component-wise distance. This defines a final $L$th fully-connected layer for the network which joins the two siamese twins.

We depict one example above (Figure 4), which shows the largest version of our model that we considered. This network also gave the best result for any network on the verification task.

> 我们的标准模型是一个孪生卷积神经网络，包含 $L$ 层，每层有 $N_l$ 个单元，其中 $h_{1,l}$ 代表第一个孪生网络的第 $l$ 层的隐藏向量，而 $h_{2,l}$ 则代表第二个孪生网络的对应隐藏向量。我们在前 $L-2$ 层中专门使用线性整流（ReLU）单元，在剩余层中使用 $S$ 型单元。
>
> 模型由一系列卷积层组成，每层都使用一个具有不同大小的滤波器和固定步长为 $1$ 的通道。卷积滤波器的数量被指定为 $16$ 的倍数，以优化性能。网络将ReLU激活函数应用于输出特征映射，然后可以选择性地使用大小为 $2$、步长为 $2$ 的最大池化。因此，每层的第 $k$ 个滤波器映射采用以下形式：
>
> $$
> \begin{aligned}
> & a_{1, m}^{(k)}=\operatorname{max-pool}\left(\max \left(0, \mathbf{W}_{l-1, l}^{(k)} \star \mathbf{h}_{1,(l-1)}+\mathbf{b}_l\right), 2\right) \\
> & a_{2, m}^{(k)}=\operatorname{max-pool}\left(\max \left(0, \mathbf{W}_{l-1, l}^{(k)} \star \mathbf{h}_{2,(l-1)}+\mathbf{b}_l\right), 2\right)
> \end{aligned}
> $$
>
> 其中，$W_{l-1,l}$ 是代表第 $l$ 层特征映射的三维张量，而 $\star$ 代表有效的卷积操作，仅返回每个卷积滤波器和输入特征映射之间完全重叠的结果输出单元。
>
> 最后一个卷积层中的单元被展平为一个单一向量。此卷积层后跟一个全连接层，然后是另一层，计算每个孪生网络之间的诱导距离度量，并将其提供给一个单一的S型输出单元。更精确地说，预测向量给出为$\mathcal{p} = \sigma\left(\sum_j \alpha_j\left|\mathbf{h}_{1, L-1}^{(j)}-\mathbf{h}_{2, L-1}^{(j)}\right|\right)$，其中$σ$是S型激活函数。最后一层在第（$L-1$）隐藏层的学习特征空间上引入了一个度量，并对两个特征向量之间的相似性进行评分。$α_j$是模型在训练过程中学习的额外参数，用于权衡分量距离的重要性。这为连接两个孪生网络的网络定义了一个最终的$L$层全连接层。
>
> 我们在上方描绘了一个示例（图4），展示了我们考虑过的最大版本的模型。该网络在验证任务中也给出了任何网络的最佳结果。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/match/Siamse/Figure3.png)

#### 3.2. Learning

**Loss function.** Let $M$ represent the minibatch size, where $i$ indexes the $i$th minibatch. Now let $y(x^{(i)}_1 , x^{(i)}_2 )$ be a length-$M$ vector which contains the labels for the minibatch, where we assume $y(x^{(i)}_1 , x^{(i)}_2) = 1$ whenever $x_1$ and $x_2$ are from the same character class and $y(x^{(i)}_1 , x^{(i)}_2 ) = 0$ otherwise. We impose a regularized cross-entropy objective on our binary classifier of the following form:
$$
\begin{gathered}
\mathcal{L}\left(x_1^{(i)}, x_2^{(i)}\right)=\mathbf{y}\left(x_1^{(i)}, x_2^{(i)}\right) \log \mathbf{p}\left(x_1^{(i)}, x_2^{(i)}\right)+ \\
\left(1-\mathbf{y}\left(x_1^{(i)}, x_2^{(i)}\right)\right) \log \left(1-\mathbf{p}\left(x_1^{(i)}, x_2^{(i)}\right)\right)+\boldsymbol{\lambda}^T|\mathbf{w}|^2
\end{gathered}
$$
**Optimization**. This objective is combined with standard backpropagation algorithm, where the gradient is additive across the twin networks due to the tied weights. We fix a minibatch size of $128$ with learning rate $η_j$ , momentum $µ_j$ , and $L_2$ regularization weights $λ_j$ defined layer-wise, so that our update rule at epoch $T$ is as follows:
$$
\begin{gathered}
\mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)=\mathbf{w}_{k j}^{(T)}+\Delta \mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)+2 \lambda_j\left|\mathbf{w}_{k j}\right| \\
\Delta \mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)=-\eta_j \nabla w_{k j}^{(T)}+\mu_j \Delta \mathbf{w}_{k j}^{(T-1)}
\end{gathered}
$$
where $∇w_{kj}$ is the partial derivative with respect to the weight between the $j$th neuron in some layer and the $k$th neuron in the successive layer.

**Weight initialization.** We initialized all network weights in the convolutional layers from a normal distribution with zero-mean and a standard deviation of $10^{−2}$ . Biases were also initialized from a normal distribution, but with mean $0.5$ and standard deviation $10^{−2}$ . In the fully-connected layers, the biases were initialized in the same way as the convolutional layers, but the weights were drawn from a much wider normal distribution with zero-mean and standard deviation $2 × 10^{−1}$ .

**Learning schedule.** Although we allowed for a different learning rate for each layer, learning rates were decayed uniformly across the network by $1$ percent per epoch, so that $η^{(T)}_j = 0.99η^{(T −1)}_j$ . We found that by annealing the learning rate, the network was able to converge to local minima more easily without getting stuck in the error surface. We fixed momentum to start at $0.5$ in every layer, increasing linearly each epoch until reaching the value $µ_j$ , the individual momentum term for the $j$th layer.

We trained each network for a maximum of 200 epochs, but monitored one-shot validation error on a set of 320 oneshot learning tasks generated randomly from the alphabets and drawers in the validation set. When the validation error did not decrease for 20 epochs, we stopped and used the parameters of the model at the best epoch according to the one-shot validation error. If the validation error continued to decrease for the entire learning schedule, we saved the final state of the model generated by this procedure.

> **损失函数。**设$M$代表小批量的大小，其中$i$代表第$i$个小批量。现在，设$y(x^{(i)}_1 , x^{(i)}_2 )$是一个长度为$M$的向量，包含小批量的标签，我们假设每当$x_1$和$x_2$来自同一字符类别时，$y(x^{(i)}_1 , x^{(i)}_2) = 1$，否则$y(x^{(i)}_1 , x^{(i)}_2 ) = 0$。我们在二元分类器上施加正则化交叉熵目标，形式如下：
> $$
> \begin{gathered}
> \mathcal{L}\left(x_1^{(i)}, x_2^{(i)}\right)=\mathbf{y}\left(x_1^{(i)}, x_2^{(i)}\right) \log \mathbf{p}\left(x_1^{(i)}, x_2^{(i)}\right)+ \\
> \left(1-\mathbf{y}\left(x_1^{(i)}, x_2^{(i)}\right)\right) \log \left(1-\mathbf{p}\left(x_1^{(i)}, x_2^{(i)}\right)\right)+\boldsymbol{\lambda}^T|\mathbf{w}|^2
> \end{gathered}
> $$
>
> **优化**。该目标与标准反向传播算法相结合，由于权重绑定，梯度在孪生网络中是可加的。我们固定小批量大小为$128$，学习率为$η_j$，动量为$µ_j$，并且逐层定义$L_2$正则化权重$λ_j$，以便我们在 epoch $T$的更新规则如下：
> $$
> \begin{gathered}
> \mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)=\mathbf{w}_{k j}^{(T)}+\Delta \mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)+2 \lambda_j\left|\mathbf{w}_{k j}\right| \\
> \Delta \mathbf{w}_{k j}^{(T)}\left(x_1^{(i)}, x_2^{(i)}\right)=-\eta_j \nabla w_{k j}^{(T)}+\mu_j \Delta \mathbf{w}_{k j}^{(T-1)}
> \end{gathered}
> $$
>
> 其中，$∇w_{kj}$是关于某一层中第$j$个神经元与后续层中第$k$个神经元之间权重的偏导数。
>
> **权重初始化。**我们从均值为零、标准差为$10^{-2}$的正态分布中初始化卷积层的所有网络权重。偏置也是从正态分布初始化的，但均值为$0.5$，标准差为$10^{-2}$。在全连接层中，偏置的初始化方式与卷积层相同，但权重是从均值为零、标准差为$2 \times 10^{-1}$的更宽的正态分布中抽取的。
>
> **学习计划。**尽管我们允许每层有不同的学习率，但学习率在整个网络中每个时代都会统一衰减 $1\%$，因此$η^{(T)}_j = 0.99η^{(T -1)}_j$。我们发现，通过学习率的退火，网络能够更容易地收敛到局部最小值，而不会陷入误差曲面。我们将动量初始值固定为每层的 $0.5$，每个时代线性增加，直到达到第$j$层的单独动量项 $µ_j$。
>
> 我们对每个网络进行了最多200个时代的训练，但我们在由验证集中的字母和抽屉随机生成的320个一次性学习任务上监测了一次性验证错误。当验证错误在20个时代内没有减少时，我们停止并使用一次性验证错误最佳时代的模型参数。如果验证错误在整个学习计划期间持续下降，我们保存了此过程生成的模型的最终状态。

**Hyperparameter optimization.** We used the beta version of Whetlab, a Bayesian optimization framework, to perform hyperparameter selection. For learning schedule and regularization hyperparameters, we set the layerwise learning rate $η_j ∈ [10^{−4} , 10^{−1} ]$, layer-wise momentum $µ_j ∈ [0, 1]$, and layer-wise L2 regularization penalty $λ_j ∈ [0, 0.1]$. For network hyperparameters, we let the size of convolutional filters vary from 3x3 to 20x20, while the number of convolutional filters in each layer varied from 16 to 256 using multiples of 16. Fully-connected layers ranged from 128 to 4096 units, also in multiples of 16. We set the optimizer to maximize one-shot validation set accuracy. The score assigned to a single Whetlab iteration was the highest value of this metric found during any epoch.

**Affine distortions.** In addition, we augmented the training set with small affine distortions (Figure 5). For each image pair $x_1, x_2$, we generated a pair of affine transformations $T_1$, $T_2$ to yield $x_1^{'} = T_1(x_1), x_2^{'} = T_2(x_2)$, where $T_1$, $T_2$ are determined stochastically by a multidimensional uniform distribution. So for an arbitrary transform $T$, we have $T = (θ, ρ_x, ρ_y, s_x, s_y, t_x, t_x)$, with $θ ∈ [−10.0, 10.0]$, $ρ_x, ρ_y ∈ [−0.3, 0.3]$, $s_x, s_y ∈ [0.8, 1.2]$, and $t_x, t_y ∈ [−2, 2]$. Each of these components of the transformation is included with probability $0.5$.

> **超参数优化**。我们使用了Whetlab的beta版本，这是一个贝叶斯优化框架，用于执行超参数选择。对于学习计划和正则化超参数，我们逐层设置学习率$η_j ∈ [10^{-4}, 10^{-1}]$，逐层动量$µ_j ∈ [0, 1]$，以及逐层L2正则化惩罚$λ_j ∈ [0, 0.1]$。对于网络超参数，我们让卷积滤波器的大小从3x3变化到20x20，而每层中的卷积滤波器数量从16到256变化，增量为16的倍数。全连接层的单元数从128到4096变化，也是16的倍数。我们设置优化器以最大化一次性验证集的准确性。分配给单个Whetlab迭代的分数是在任何时代中找到的此指标的最高值。
>
> **仿射扭曲**。此外，我们用小的仿射扭曲增强了训练集（图5）。对于每对图像$x_1, x_2$，我们生成了一对仿射变换$T_1, T_2$，以产生$x_1^{'}= T_1(x_1), x_2^{'}= T_2(x_2)$，其中$T_1, T_2$由多维均匀分布随机确定。因此，对于任意的变换$T$，我们有$T = (θ, ρ_x, ρ_y, s_x, s_y, t_x, t_x)$，其中$θ ∈ [−10.0, 10.0]$，$ρ_x, ρ_y ∈ [−0.3, 0.3]$，$s_x, s_y ∈ [0.8, 1.2]$，和$t_x, t_y ∈ [−2, 2]$。变换的每个组成部分都以0.5的概率包含在内。

## 4. Experiments

We trained our model on a subset of the Omniglot data set, which we first describe. We then provide details with respect to verification and one-shot performance.

#### 4.1. The Omniglot Dataset

The Omniglot data set was collected by Brenden Lake and his collaborators at MIT via Amazon’s Mechanical Turk to produce a standard benchmark for learning from few examples in the handwritten character recognition domain (Lake et al., 2011).1 Omniglot contains examples from 50 alphabets ranging from well-established international languages like Latin and Korean to lesser known local dialects. It also includes some fictitious character sets such as Aurek-Besh and Klingon (Figure 6).

The number of letters in each alphabet varies considerably from about 15 to upwards of 40 characters. All characters across these alphabets are produced a single time by each of 20 drawers Lake split the data into a 40 alphabet background set and a 10 alphabet evaluation set. We preserve these two terms in order to distinguish from the normal training, validation, and test sets that can be generated from the background set in order to tune models for verification. The background set is used for developing a model by learning hyperparameters and feature mappings. Conversely, the evaluation set is used only to measure the one-shot classification performance.

> Omniglot数据集由Brenden Lake及其麻省理工学院的合作者通过亚马逊的Mechanical Turk收集，旨在为手写字符识别领域的小样本学习制定一个标准的基准（Lake等人，2011）。Omniglot包含了来自50种字母表的例子，从广为人知的国际语言（如拉丁语和韩语）到鲜为人知的地方方言。它还包含了一些虚构的字符集，如Aurek-Besh和Klingon（图6）。
>
> 每种字母表中的字母数量差异很大，从大约15个字符到40多个字符不等。这些字母表中的所有字符都由20个抽屉中的每一个单独制作一次。Lake将数据分为40个字母表的背景集和10个字母表的评估集。我们保留这两个术语，以便与可以从背景集中生成的正常训练、验证和测试集区分开来，以便调整模型以进行验证。背景集通过学习超参数和特征映射来开发模型。相反，评估集仅用于衡量一次性分类性能。

#### 4.2. Verification

To train our verification network, we put together three different data set sizes with 30,000, 90,000, and 150,000 training examples by sampling random same and different pairs. We set aside sixty percent of the total data for training: 30 alphabets out of 50 and 12 drawers out of 20.

We fixed a uniform number of training examples per alphabet so that each alphabet receives equal representation during optimization, although this is not guaranteed to the individual character classes within each alphabet. By adding affine distortions, we also produced an additional copy of the data set corresponding to the augmented version of each of these sizes. We added eight transforms for each training example, so the corresponding data sets have 270,000, 810,000, and 1,350,000 effective examples.

To monitor performance during training, we used two strategies. First, we created a validation set for verification with 10,000 example pairs taken from 10 alphabets and 4 additional drawers. We reserved the last 10 alphabets and 4 drawers for testing, where we constrained these to be the same ones used in Lake et al. (Lake et al., 2013). Our other strategy leveraged the same alphabets and drawers to generate a set of 320 one-shot recognition trials for the validation set which mimic the target task on the evaluation set. In practice, this second method of determining when to stop was at least as effective as the validation error for the verification task so we used it as our termination criterion.

In the table below (Table 1), we list the final verification results for each of the six possible training sets, where the listed test accuracy is reported at the best validation checkpoint and threshold. We report results across six different training runs, varying the training set size and toggling distortions.

In Figure 7, we have extracted the first 32 filters from both of our top two performing networks on the verification task, which were trained on the 90k and 150k data sets with affine distortions and the architecture shown in Figure 3. While there is some co-adaptation between filters, it is easy to see that some of the filters have assumed different roles with respect to the original input space.

> 为了训练我们的验证网络，我们通过随机采样相同和不同的对来组合了三个不同大小的数据集，分别有30,000、90,000和150,000个训练样本。我们留出总数据的百分之六十用于训练：50个字母表中的30个，以及20个抽屉中的12个。
>
> 我们为每个字母表固定了统一的训练样本数量，以确保在优化过程中每个字母表都能得到平等的表示，尽管这并不能保证每个字母表内的各个字符类都能得到平等的表示。通过添加仿射扭曲，我们还为每种大小的增强版本生成了一个额外的数据集副本。我们为每个训练样本添加了八种变换，因此相应的数据集具有270,000、810,000和1,350,000个有效样本。
>
> 为了监控训练过程中的性能，我们采用了两种策略。首先，我们从10个字母表和另外4个抽屉中抽取了10,000个样本对，创建了一个用于验证的验证集。我们保留了最后10个字母表和4个抽屉用于测试，并限制这些与Lake等人的研究中使用的相同（Lake等人，2013）。我们的另一种策略是利用相同的字母表和抽屉为验证集生成了一组320个一次性识别试验，这些试验模拟了评估集上的目标任务。在实践中，确定何时停止的第二种方法至少与验证任务的验证错误一样有效，因此我们将其用作终止标准。
>
> 在下表（表1）中，我们列出了六个可能的训练集中每个的最终验证结果，其中列出的测试准确度是在最佳验证检查点和阈值处报告的。我们报告了六次不同训练运行的结果，这些运行改变了训练集的大小并切换了扭曲。
>
> 在图7中，我们从验证任务中表现最好的两个网络中提取了前32个滤波器，这两个网络是在带有仿射扭曲的90k和150k数据集上训练的，其架构如图3所示。虽然滤波器之间存在一些共同适应，但很容易看出，有些滤波器在原始输入空间中扮演了不同的角色。

#### 4.3. One-shot Learning

Once we have optimized a siamese network to master the verification task, we are ready to demonstrate the discriminative potential of our learned features at one-shot learning. Suppose we are given a test image x, some column vector which we wish to classify into one of C categories. We are also given some other images ${x_c}^{C}_{c=1}$, a set of column vectors representing examples of each of those $C$ categories. We can now query the network using $x$, $x_c$ as our input for a range of $c = 1, . . . , C.^2$ Then predict the class corresponding to the maximum similarity.
$$
C^*=\operatorname{argmax}_c \mathbf{p}^{(c)}
$$
To empirically evaluate one-shot learning performance, Lake developed a 20-way within-alphabet classification task in which an alphabet is first chosen from among those reserved for the evaluation set, along with twenty characters taken uniformly at random. Two of the twenty drawers are also selected from among the pool of evaluation drawers. These two drawers then produce a sample of the twenty characters. Each one of the characters produced by the first drawer are denoted as test images and individually compared against all twenty characters from the second drawer, with the goal of predicting the class corresponding to the test image from among all of the second drawer’s characters. An individual example of a one-shot learning trial is depicted in Figure 7. This process is repeated twice for all alphabets, so that there are 40 one-shot learning trials for each of the ten evaluation alphabets. This constitutes a total of 400 one-shot learning trials, from which the classification accuracy is calculated.

The one-shot results are given in Table 2. We borrow the baseline results from (Lake et al., 2013) for comparison to our method. We also include results from a nonconvolutional siamese network with two fully-connected layers.

At 92 percent our convolutional method is stronger than any model except HBPL itself. which is only slightly behind human error rates. While HBPL exhibits stronger results overall, our top-performing convolutional network did not include any extra prior knowledge about characters or strokes such as generative information about the drawing process. This is the primary advantage of our model.

> 一旦我们优化了一个暹罗网络以掌握验证任务，我们就准备好展示我们学习到的特征在一次性学习中的判别潜力。假设我们得到了一张测试图像x，我们希望将其分类为 $C$ 个类别中的某一类的某个列向量。我们还得到了其他一些图像${x_c}^{C}_{c=1}$，这是一组代表这些$C$类中每一个的例子的列向量。现在，我们可以使用$x$，$x_c$作为输入来查询网络，范围为$c = 1,...,C.^2$。然后预测与最大相似度对应的类别。
>
> $$
> C^*=\operatorname{argmax}_c \mathbf{p}^{(c)}
> $$
>
> 为了实证评估一次性学习的性能，Lake开发了一个20路字母内分类任务，其中首先从为评估集保留的字母中选择一个字母，同时随机均匀选择二十个字符。还从评估抽屉池中选择了两个抽屉。然后，这两个抽屉生成了二十个字符的样本。由第一个抽屉生成的每个字符都被标记为测试图像，并与第二个抽屉中的所有二十个字符进行逐个比较，目标是从第二个抽屉的所有字符中预测与测试图像对应的类别。图7展示了一次性学习试验的一个单独示例。这个过程对所有字母重复两次，因此十个评估字母中的每一个都有40个一次性学习试验。这总共构成了400个一次性学习试验，从中计算出分类精度。
>
> 一次性结果如表2所示。我们借用（Lake等人，2013）的基线结果与我们的方法进行比较。我们还包括了具有两个全连接层的非卷积暹罗网络的结果。
>
> 我们的卷积方法准确率为92%，强于除HBPL本身以外的任何模型，仅略低于人类错误率。虽然HBPL总体上表现出更强的结果，但我们表现最佳的卷积网络没有包含任何关于字符或笔划的额外先验知识，如绘图过程的生成信息。这是我们模型的主要优势。

#### 4.4. MNIST One-shot Trial

The Omniglot data set contains a small handful of samples for every possible class of letter; for this reason, the original authors refer to it as a sort of “MNIST transpose”, where the number of classes far exceeds the number of training instances (Lake et al., 2013). We thought it would be interesting to monitor how well a model trained on Omniglot can generalize to MNIST, where we treat the 10 digits in MNIST as an alphabet and then evaluate a 10-way oneshot classification task. We followed a similar procedure to Omniglot, generating 400 one-shot trials on the MNIST test set, but excluding any fine tuning on the training set. All 28x28 images were upsampled to 35x35, then given to a reduced version of our model trained on 35x35 images from Omniglot which were downsampled by a factor of 3. We also evaluated the nearest-neighbor baseline on this task.

Table 3 shows the results from this experiment. The nearest neighbor baseline provides similar performance to Omniglot, while the performance of the convolutional network drops by a more significant amount. However, we are still able to achieve reasonable generalization from the features learned on Ominglot without training at all on MNIST.

> Omniglot数据集包含每个可能字母类别的少量样本；因此，原始作者将其称为一种“MNIST转置”，其中类别数量远远超过训练实例数量（Lake等人，2013）。我们认为监测在Omniglot上训练的模型能否很好地推广到MNIST会很有趣，我们将MNIST中的10个数字视为一个字母表，然后评估一个10路一次性分类任务。我们遵循与Omniglot类似的程序，在MNIST测试集上生成400个一次性试验，但不包括对训练集的任何微调。所有28x28图像都被上采样到35x35，然后提供给在Omniglot的35x35图像上训练的简化版模型，这些图像被下采样了3倍。我们还评估了此任务上的最近邻基线。
>
> 表3显示了此实验的结果。最近邻基线提供了与Omniglot相似的性能，而卷积网络的性能下降幅度更大。但是，我们仍然能够从在Ominglot上学到的特征中实现合理的泛化，而根本无需在MNIST上进行训练。

## 5. Conclusions

We have presented a strategy for performing one-shot classification by first learning deep convolutional siamese neural networks for verification. We outlined new results comparing the performance of our networks to an existing state-of-the-art classifier developed for the Omniglot data set. Our networks outperform all available baselines by a significant margin and come close to the best numbers achieved by the previous authors. We have argued that the strong performance of these networks on this task indicate not only that human-level accuracy is possible with our metric learning approach, but that this approach should extend to one-shot learning tasks in other domains, especially for image classification.

In this paper, we only considered training for the verification task by processing image pairs and their distortions using a global affine transform. We have been experimenting with an extended algorithm that exploits the data about the individual stroke trajectories to produce final computed distortions (Figure 8). By imposing local affine transformations on the strokes and overlaying them into a composite image, we are hopeful that we can learn features which are better adapted to the variations that are commonly seen in new examples.

> 我们提出了一种策略，即首先学习深度卷积暹罗神经网络进行验证，以执行一次性分类。我们概述了新的结果，将我们的网络与为Omniglot数据集开发的现有最先进的分类器的性能进行了比较。我们的网络在所有可用的基线中都表现出色，并接近之前作者获得的最佳数据。我们认为，这些网络在此任务上的出色表现不仅表明我们的度量学习方法可以达到人类级别的精度，而且这种方法应扩展到其他领域的一次性学习任务，特别是图像分类。
>
> 在本文中，我们只考虑通过处理图像对及其使用全局仿射变换产生的扭曲来进行验证任务的训练。我们一直在试验一种扩展算法，该算法利用有关单个笔划轨迹的数据来产生最终的计算扭曲（图8）。通过对笔划施加局部仿射变换并将它们叠加到合成图像中，我们希望我们能够学习到更适应新示例中常见变化的特征。