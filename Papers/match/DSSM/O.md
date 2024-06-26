# Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

## ABSTRACT

Latent semantic models, such as LSA, intend to map a query to its relevant documents at the semantic level where keyword-based matching often fails. In this study we strive to develop a series of new latent semantic models with a deep structure that project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them. The proposed deep structured semantic models are discriminatively trained by maximizing the conditional likelihood of the clicked documents given a query using the clickthrough data. To make our models applicable to large-scale Web search applications, we also use a technique called word hashing, which is shown to effectively scale up our semantic models to handle large vocabularies which are common in such tasks. The new models are evaluated on a Web document ranking task using a real-world data set. Results show that our best model significantly outperforms other latent semantic models, which were considered state-of-the-art in the performance prior to the work presented in this paper.

> 潜在语义模型，如LSA，旨在将查询映射到语义层面上的相关文档，这是基于关键词的匹配常常失败的地方。在这项研究中，我们努力开发一系列具有深层结构的新潜在语义模型，将查询和文档投影到一个共同的低维空间中，其中给定查询的文档的相关性可以很容易地计算为它们之间的距离。所提出的深度结构语义模型是通过最大化点击文档的条件似然性来判别训练的，给定查询使用点击数据。为了使我们的模型适用于大规模的网络搜索应用，我们还使用了一种称为词哈希的技术，该技术被证明可以有效地扩展我们的语义模型，以处理此类任务中常见的大量词汇。新模型在真实世界数据集上的网络文档排名任务中进行了评估。结果表明，我们最好的模型显著优于其他潜在语义模型，这些模型在本文介绍的工作之前的性能被认为是最先进的。

## 1. INTRODUCTION

Modern search engines retrieve Web documents mainly by matching keywords in documents with those in search queries. However, lexical matching can be inaccurate due to the fact that a concept is often expressed using different vocabularies and language styles in documents and queries.

Latent semantic models such as latent semantic analysis (LSA) are able to map a query to its relevant documents at the semantic level where lexical matching often fails (e.g., [6][15][2][8][21]). These latent semantic models address the language discrepancy between Web documents and search queries by grouping different terms that occur in a similar context into the same semantic cluster. Thus, a query and a document, represented as two vectors in the lower-dimensional semantic space, can still have a high similarity score even if they do not share any term. Extending from LSA, probabilistic topic models such as probabilistic LSA (PLSA) and Latent Dirichlet Allocation (LDA) have also been proposed for semantic matching [15][2]. However, these models are often trained in an unsupervised manner using an objective function that is only loosely coupled with the evaluation metric for the retrieval task. Thus the performance of these models on Web search tasks is not as good as originally expected.

Recently, two lines of research have been conducted to extend the aforementioned latent semantic models, which will be briefly reviewed below.

First, clickthrough data, which consists of a list of queries and their clicked documents, is exploited for semantic modeling so as to bridge the language discrepancy between search queries and Web documents [9][10]. For example, Gao et al. [10] propose the use of Bi-Lingual Topic Models (BLTMs) and linear Discriminative Projection Models (DPMs) for query-document matching at the semantic level. These models are trained on clickthrough data using objectives that tailor to the document ranking task. More specifically, BLTM is a generative model that requires that a query and its clicked documents not only share the same distribution over topics but also contain similar factions of words assigned to each topic. In contrast, the DPM is learned using the S2Net algorithm [26] that follows the pairwise learning- to-rank paradigm outlined in [3]. After projecting term vectors of queries and documents into concept vectors in a low-dimensional semantic space, the concept vectors of the query and its clicked documents have a smaller distance than that of the query and its unclicked documents. Gao et al. [10] report that both BLTM and DPM outperform significantly the unsupervised latent semantic models, including LSA and PLSA, in the document ranking task. However, the training of BLTM, though using clickthrough data, is to maximize a log-likelihood criterion which is sub-optimal for the evaluation metric for document ranking. On the other hand, the training of DPM involves large-scale matrix multiplications. The sizes of these matrices often grow quickly with the vocabulary size, which could be of an order of millions in Web search tasks. In order to make the training time tolerable, the vocabulary was pruned aggressively. Although a small vocabulary makes the models trainable, it leads to suboptimal performance.

> 现代搜索引擎主要通过将文档中的关键词与搜索查询中的关键词进行匹配来检索Web文档。然而，由于一个概念在文档和查询中通常使用不同的词汇和语言风格来表达，因此词汇匹配可能不准确。
>
> 潜在语义模型，如潜在语义分析（LSA），能够在词汇匹配经常失败的情况下，在语义层面上将查询映射到其相关文档（例如，[6][15][2][8][21]）。这些潜在语义模型通过将出现在相似上下文中的不同术语分组到相同的语义簇中，来解决Web文档和搜索查询之间的语言差异。因此，即使查询和文档在较低的语义空间中表示为两个向量，它们也可能具有高分数的相似性，即使它们没有共享任何术语。从LSA扩展，还提出了概率主题模型，如概率LSA（PLSA）和隐式狄利克雷分配（LDA），用于语义匹配[15][2]。然而，这些模型通常使用与目标函数松散耦合的评价指标以无监督方式进行训练，该目标函数与检索任务的评价指标松散耦合。因此，这些模型在Web搜索任务上的性能并不如最初预期的那样好。
>
> 最近，进行了两条研究路线来扩展上述潜在语义模型，下面将简要回顾。
>
> 首先，点击数据，由查询及其点击文档列表组成，被用于语义建模，以便弥合搜索查询和Web文档之间的语言差异[9][10]。例如，Gao等人[10]提出使用双语主题模型（BLTMs）和线性判别投影模型（DPMs）在语义级别进行查询-文档匹配。这些模型使用针对文档排序任务的目标在点击数据上进行训练。更具体地说，BLTM是一个生成模型，要求查询及其点击文档不仅共享相同的主题分布，而且每个主题分配的单词也具有相似的比例。相比之下，DPM使用S2Net算法[26]进行学习，该算法遵循[3]中概述的成对学习排名范式。在将查询和文档的术语向量投影到低维语义空间中的概念向量后，查询及其点击文档的概念向量之间的距离小于查询及其未点击文档之间的距离。Gao等人[10]报告说，在文档排名任务中，BLTM和DPM均显著优于无监督的潜在语义模型，包括LSA和PLSA。然而，尽管使用点击数据训练BLTM，但最大化对数似然准则对于文档排名评估指标来说并不是最优的。另一方面，DPM的训练涉及大规模矩阵乘法。这些矩阵的大小通常随着词汇表的大小迅速增长，在Web搜索任务中可能达到数百万个。为了使训练时间可承受，词汇表被积极修剪。虽然小词汇表使模型可训练，但它导致次优性能。

In the second line of research, Salakhutdinov and Hinton extended the semantic modeling using deep auto-encoders [22].

They demonstrated that hierarchical semantic structure embedded in the query and the document can be extracted via deep learning. Superior performance to the conventional LSA is reported [22]. However, the deep learning approach they used still adopts an unsupervised learning method where the model parameters are optimized for the reconstruction of the documents rather than for differentiating the relevant documents from the irrelevant ones for a given query. As a result, the deep learning models do not significantly outperform the baseline retrieval models based on keyword matching. Moreover, the semantic hashing model also faces the scalability challenge regarding large-scale matrix multiplication. We will show in this paper that the capability of learning semantic models with large vocabularies is crucial to obtain good results in real-world Web search tasks.

In this study, extending from both research lines discussed above, we propose a series of Deep Structured Semantic Models (DSSM) for Web search. More specifically, our best model uses a deep neural network (DNN) to rank a set of documents for a given query as follows. First, a non-linear projection is performed to map the query and the documents to a common semantic space. Then, the relevance of each document given the query is calculated as the cosine similarity between their vectors in that semantic space. The neural network models are discriminatively trained using the clickthrough data such that the conditional likelihood of the clicked document given the query is maximized. Different from the previous latent semantic models that are learned in an unsupervised fashion, our models are optimized directly for Web document ranking, and thus give superior performance, as we will show shortly. Furthermore, to deal with large vocabularies, we propose the so-called word hashing method, through which the high-dimensional term vectors of queries or documents are projected to low-dimensional letter based n-gram vectors with little information loss. In our experiments, we show that, by adding this extra layer of representation in semantic models, word hashing enables us to learn discriminatively the semantic models with large vocabularies, which are essential for Web search. We evaluated the proposed DSSMs on a Web document ranking task using a real-world data set. The results show that our best model outperforms all the competing methods with a significant margin of 2.5-4.3% in NDCG@1.

In the rest of the paper, Section 2 reviews related work. Section 3 describes our DSSM for Web search. Section 4 presents the experiments, and Section 5 concludes the paper.

> 在第二条研究路线上，Salakhutdinov和Hinton使用深度自动编码器扩展了语义建模[22]。
>
> 他们展示了可以通过深度学习提取查询和文档中嵌入的层次语义结构。据报告，其性能优于传统的LSA[22]。然而，他们使用的深度学习方法仍然采用无监督学习方法，其中模型参数是为了重建文档而不是为了区分给定查询的相关文档和不相关文档而优化的。因此，深度学习模型在基于关键词匹配的基线检索模型上并没有显著的优势。此外，语义哈希模型也面临着大规模矩阵乘法的可扩展性挑战。我们将在本文中展示，学习具有大量词汇的语义模型的能力对于在现实世界的Web搜索任务中获得良好结果至关重要。
>
> 在本研究中，我们从上述两条研究路线出发，提出了一系列用于Web搜索的深度结构语义模型（DSSM）。更具体地说，我们的最佳模型使用深度神经网络（DNN）对给定查询的一组文档进行排序，方法如下。首先，执行非线性投影以将查询和文档映射到共同的语义空间。然后，给定查询的每个文档的相关性计算为该语义空间中它们向量之间的余弦相似度。神经网络模型使用点击数据以判别方式进行训练，以便最大化给定查询的点击文档的条件似然性。与我们之前以无监督方式学习的潜在语义模型不同，我们的模型直接针对Web文档排名进行了优化，因此提供了卓越的性能，正如我们将要展示的。此外，为了处理大量词汇，我们提出了所谓的词哈希方法，通过该方法，查询或文档的高维术语向量被投影到基于字母的n-gram低维向量上，几乎没有信息损失。在我们的实验中，我们展示了通过在语义模型中添加这一额外的表示层，词哈希使我们能够学习具有大量词汇的判别语义模型，这对于Web搜索至关重要。我们在一个Web文档排名任务上评估了所提出的DSSM，使用了真实世界的数据集。结果表明，我们的最佳模型在NDCG@1指标上显著优于所有竞争方法，优势幅度为2.5-4.3%。
>
> 在本文的其余部分，第2节回顾了相关工作。第3节描述了我们的Web搜索DSSM。第4节介绍了实验，第5节总结了论文。

## 2 RELATED WORK

Our work is based on two recent extensions to the latent semantic models for IR. The first is the exploration of the clickthrough data for learning latent semantic models in a supervised fashion [10]. The second is the introduction of deep learning methods for semantic modeling [22].

> 我们的工作基于信息检索中潜在语义模型的两种最新扩展。第一种是探索点击数据以监督方式学习潜在语义模型[10]。第二种是引入深度学习方法进行语义建模[22]。

#### 2.1 Latent Semantic Models and the Use of Clickthrough Data

The use of latent semantic models for query-document matching is a long-standing research topic in the IR community. Popular models can be grouped into two categories, linear projection models and generative topic models, which we will review in turn.

The most well-known linear projection model for IR is LSA [6]. By using the singular value decomposition (SVD) of a document-term matrix, a document (or a query) can be mapped to a low-dimensional concept vector $
\widehat{\mathbf{D}}=\mathbf{A}^T \mathbf{D}
$ , where the $\mathbf{A}$ is the projection matrix. In document search, the relevance score between a query and a document, represented respectively by term vectors  $\mathbf{Q}$ and $\mathbf{D}$, is assumed to be proportional to their cosine similarity score of the corresponding concept vectors $\widehat{\mathbf{Q}}$ and $\widehat{\mathbf{D}}$ , according to the projection matrix $\mathbf{A}$
$$
\operatorname{sim}_{\mathbf{A}}(\mathbf{Q}, \mathbf{D})=\frac{\widehat{\mathbf{Q}}^T \widehat{\mathbf{D}}}{\|\widehat{\mathbf{Q}}\|\|\widehat{\mathbf{D}}\|}
$$
In addition to latent semantic models, the translation models trained on clicked query-document pairs provide an alternative approach to semantic matching [9]. Unlike latent semantic models, the translation-based approach learns translation relationships directly between a term in a document and a term in a query. Recent studies show that given large amounts of clickthrough data for training, this approach can be very effective [9][10]. We will also compare our approach with translation models experimentally as reported in Section 4.

> 在IR社区中，使用潜在语义模型进行 query-document 匹配是一个长期的研究课题。流行的模型可以分为两类：线性投影模型和生成式主题模型，我们将依次进行回顾。
>
> IR中最著名的线性投影模型是LSA[6]。通过使用 document-term 矩阵的奇异值分解（SVD），document（或query）可以被映射到一个低维的概念向量 $\widehat{\mathbf{D}}=\mathbf{A}^T \mathbf{D}$，其中$\mathbf{A}$是投影矩阵。在文档搜索中，根据投影矩阵$\mathbf{A}$，Query和document之间的相关性得分（分别由术语向量$\mathbf{Q}$和$\mathbf{D}$表示）被认为与它们对应的概念向量 $\widehat{\mathbf{Q}}$ 和 $\widehat{\mathbf{D}}$ 的余弦相似度得分成正比。
>
> $$
> \operatorname{sim}_{\mathbf{A}}(\mathbf{Q}, \mathbf{D})=\frac{\widehat{\mathbf{Q}}^T \widehat{\mathbf{D}}}{\|\widehat{\mathbf{Q}}\|\|\widehat{\mathbf{D}}\|}
> $$
>
> 除了潜在语义模型外，基于 clicked query-document 对训练的翻译模型也为语义匹配提供了一种替代方法[9]。与潜在语义模型不同，基于翻译的方法直接在文档中的一个术语和查询中的一个术语之间学习翻译关系。最近的研究表明，给定大量的点击数据进行训练，这种方法可以非常有效[9][10]。我们也将在第4节中通过实验将我们的方法与翻译模型进行比较。

#### 2.2 Deep Learning

Recently, deep learning methods have been successfully applied to a variety of language and information retrieval applications [1][4][7][19][22][23][25]. By exploiting deep architectures, deep learning techniques are able to discover from training data the hidden structures and features at different levels of abstractions useful for the tasks. In [22] Salakhutdinov and Hinton extended the LSA model by using a deep network (auto-encoder) to discover the hierarchical semantic structure embedded in the query and the document. They proposed a semantic hashing (SH) method which uses bottleneck features learned from the deep auto-encoder for information retrieval. These deep models are learned in two stages. First, a stack of generative models (i.e., the restricted Boltzmann machine) are learned to map layer-by-layer a term vector representation of a document to a low-dimensional semantic concept vector. Second, the model parameters are finetuned so as to minimize the cross entropy error between the original term vector of the document and the reconstructed term vector. The intermediate layer activations are used as features (i.e., bottleneck) for document ranking. Their evaluation shows that the SH approach achieves a superior document retrieval performance to the LSA. However, SH suffers from two problems, and cannot outperform the standard lexical matching based retrieval model (e.g., cosine similarity using TF-IDF term weighting). The first problem is that the model parameters are optimized for the re-construction of the document term vectors rather than for differentiating the relevant documents from the irrelevant ones for a given query. Second, in order to make the computational cost manageable, the term vectors of documents consist of only the most-frequent 2000 words. In the next section, we will show our solutions to these two problems.

> 最近，深度学习方法已成功应用于各种语言和信息检索应用[1][4][7][19][22][23][25]。通过利用深度架构，深度学习技术能够从训练数据中发现对任务有用的不同抽象级别的隐藏结构和特征。在[22]中，Salakhutdinov和Hinton通过使用深度网络（自动编码器）扩展了LSA模型，以发现嵌入在查询和文档中的层次语义结构。他们提出了一种语义哈希（SH）方法，该方法使用从深度自动编码器中学习的瓶颈特征进行信息检索。这些深度模型的学习分为两个阶段。首先，学习一组生成模型（即受限玻尔兹曼机）以逐层将文档的术语向量表示映射到低维语义概念向量。其次，对模型参数进行微调，以最小化文档原始术语向量和重构术语向量之间的交叉熵误差。中间层的激活用作文档排名的特征（即瓶颈）。他们的评估表明，SH方法在文档检索性能上优于LSA。然而，SH存在两个问题，无法超越基于标准词汇匹配的检索模型（例如，使用TF-IDF术语权重的余弦相似性）。第一个问题是模型参数是针对文档术语向量的重构进行优化的，而不是针对给定查询区分相关文档和不相关文档。第二个问题是，为了使计算成本可控，文档的术语向量仅包含最常用的2000个词。在下一节中，我们将展示我们对这两个问题的解决方案。

## 3. DEEP STRUCTURED SEMANTIC MODELS FOR WEB SEARCH

#### 3.1 DNN for Computing Semantic Features

The typical DNN architecture we have developed for mapping the raw text features into the features in a semantic space is shown in Fig. 1. The input (raw text features) to the DNN is a highdimensional term vector, e.g., raw counts of terms in a query or a document without normalization, and the output of the DNN is a concept vector in a low-dimensional semantic feature space. This DNN model is used for Web document ranking as follows: 1) to map term vectors to their corresponding semantic concept vectors; 2) to compute the relevance score between a document and a query as cosine similarity of their corresponding semantic concept vectors; rf. Eq. (3) to (5).

More formally, if we denote $x$ as the input term vector, $y$ as the output vector, $l_i, i=1, \ldots, N-1$ as the intermediate hidden layers, $W_i$ as the $i$-th weight matrix, and $b_i$ as the $i$-th bias term, we have
$$
\begin{gathered}
l_1=W_1 x \\
l_i=f\left(W_i l_{i-1}+b_i\right), i=2, \ldots, N-1 \\
y=f\left(W_N l_{N-1}+b_N\right)
\end{gathered}
$$
where we use the $tanh$ as the activation function at the output layer and the hidden layers $l_i, i=2, \ldots, N-1$ :
$$
f(x)=\frac{1-e^{-2 x}}{1+e^{-2 x}}
$$
The semantic relevance score between a query $Q$ and a document $D$ is then measured as:
$$
\
R(Q, D)=\operatorname{cosine}\left(y_Q, y_D\right)=\frac{y_Q{ }^T y_D}{\left\|y_Q\right\|\left\|y_D\right\|}
$$
where $y_Q$ and $y_D$ are the concept vectors of the query and the document, respectively. In Web search, given the query, the documents are sorted by their semantic relevance scores.

Conventionally, the size of the term vector, which can be viewed as the raw bag-of-words features in IR, is identical to that of the vocabulary that is used for indexing the Web document collection. The vocabulary size is usually very large in real-world Web search tasks. Therefore, when using term vector as the input, the size of the input layer of the neural network would be unmanageable for inference and model training. To address this problem, we have developed a method called “word hashing” for the first layer of the DNN, as indicated in the lower portion of Figure 1. This layer consists of only linear hidden units in which the weight matrix of a very large size is not learned. In the following section, we describe the word hashing method in detail.

> 我们为将原始文本特征映射到语义空间中的特征而开发的典型深度神经网络（DNN）架构如图1所示。DNN的输入（原始文本特征）是高维术语向量，例如，查询或文档中的术语原始计数（未进行归一化），DNN的输出是低维语义特征空间中的概念向量。该DNN模型用于Web文档排名的方式如下：1）将术语向量映射到其对应的语义概念向量；2）根据它们对应的语义概念向量的余弦相似度计算 d 与 q 之间的相关性得分；参考公式（3）至（5）。
>
> 更正式地说，如果我们用 $x$ 表示输入术语向量，$y$ 表示输出向量，$l_i, i=1, \ldots, N-1$表示中间隐藏层，$W_i$表示第 $i$ 个权重矩阵，$b_i$ 表示第 $i$ 个偏置项，则我们有
>
> $$
> \begin{gathered}
> l_1=W_1 x \\
> l_i=f\left(W_i l_{i-1}+b_i\right), i=2, \ldots, N-1 \\
> y=f\left(W_N l_{N-1}+b_N\right)
> \end{gathered}
> $$
>
> 其中，我们在输出层和隐藏层 $l_i, i=2, \ldots, N-1$ 使用 $tanh$ 作为激活函数：
>
> $$
> f(x)=\frac{1-e^{-2 x}}{1+e^{-2 x}}
> $$
>
> 然后，查询 $Q$ 和文档 $D$ 之间的语义相关性得分测量为：
>
> $$
> R(Q, D)=\operatorname{cosine}\left(y_Q, y_D\right)=\frac{y_Q{ }^T y_D}{\left\|y_Q\right\|\left\|y_D\right\|}
> $$
>
> 其中，$y_Q$ 和 $y_D$ 分别是查询和文档的概念向量。在Web搜索中，给定查询，文档会按其语义相关性得分进行排序。
>
> 传统上，术语向量的大小，可以视为IR中的原始词袋特征，与用于索引Web文档集合的词汇表大小相同。在现实世界的Web搜索任务中，词汇表大小通常非常大。因此，当使用术语向量作为输入时，神经网络输入层的大小对于推理和模型训练来说将是不可管理的。为了解决这个问题，我们为DNN的第一层开发了一种称为“词哈希”的方法，如图1的下半部分所示。这一层仅包含线性隐藏单元，其中不会学习非常大的权重矩阵。在下一节中，我们将详细描述词哈希方法。

![Fig1](/Users/anmingyu/Github/Gor-rok/Papers/match/DSSM/Fig1.png)

#### 3.2 Word Hashing

The word hashing method described here aim to reduce the dimensionality of the bag-of-words term vectors. It is based on letter n-gram, and is a new method developed especially for our task. Given a word (e.g. good), we first add word starting and ending marks to the word (e.g. #good#). Then, we break the word into letter n-grams (e.g. letter trigrams: #go, goo, ood, od#). Finally, the word is represented using a vector of letter n-grams. 

One problem of this method is collision, i.e., two different words could have the same letter n-gram vector representation. Table 1 shows some statistics of word hashing on two vocabularies. Compared with the original size of the one-hot vector, word hashing allows us to represent a query or a document using a vector with much lower dimensionality. Take the 40Kword vocabulary as an example. Each word can be represented by a 10,306-dimentional vector using letter trigrams, giving a fourfold dimensionality reduction with few collisions. The reduction of dimensionality is even more significant when the technique is applied to a larger vocabulary. As shown in Table 1, each word in the 500K-word vocabulary can be represented by a 30,621 dimensional vector using letter trigrams, a reduction of 16-fold in dimensionality with a negligible collision rate of 0.0044% (22/500,000). 

While the number of English words can be unlimited, the number of letter n-grams in English (or other similar languages) is often limited. Moreover, word hashing is able to map the morphological variations of the same word to the points that are close to each other in the letter n-gram space. More importantly, while a word unseen in the training set always cause difficulties in word-based representations, it is not the case where the letter ngram based representation is used. The only risk is the minor representation collision as quantified in Table 1. Thus, letter ngram based word hashing is robust to the out-of-vocabulary problem, allowing us to scale up the DNN solution to the Web search tasks where extremely large vocabularies are desirable. We will demonstrate the benefit of the technique in Section 4.

In our implementation, the letter n-gram based word hashing can be viewed as a fixed (i.e., non-adaptive) linear transformation, through which an term vector in the input layer is projected to a letter n-gram vector in the next layer higher up, as shown in Figure 1. Since the letter n-gram vector is of a much lower dimensionality, DNN learning can be carried out effectively. 

> 这里描述的词哈希方法旨在降低词袋术语向量的维度。它基于字母n-gram，是专门为我们的任务开发的一种新方法。给定一个词（例如“good”），我们首先在词的开头和结尾添加标记（例如“#good#”）。然后，我们将词拆分为字母n-gram（例如字母三元组：#go，goo，ood，od#）。最后，使用字母n-gram向量来表示该词。
>
> 这种方法的一个问题是冲突，即两个不同的词可能有相同的字母n-gram向量表示。表1展示了在两个词汇表上应用词哈希的一些统计结果。与原始独热向量的大小相比，词哈希允许我们使用维度更低的向量来表示 q 或 doc。以4万个词的词汇表为例，每个词可以使用 trigram 表示为一个 10306 维的向量，维度降低了四倍，且冲突很少。当将该技术应用于更大的词汇表时，维度降低更为显著。如表1所示，使用 trigram，50万个词的词汇表中的每个词都可以表示为一个 30621 维的向量，维度降低了16倍，且冲突率可忽略不计，仅为0.0044%（22/500,000）。
>
> 虽然英语单词的数量可以无限，但英语（或其他类似语言）中的字母n-gram数量通常有限。此外，词哈希能够将同一词的形态变化映射到字母n-gram空间中彼此接近的点。更重要的是，虽然训练集中未出现的词总是会给基于词的表示带来困难，但在使用基于字母n-gram的表示时并非如此。唯一的风险是如表1所示的微小表示冲突。因此，基于字母n-gram的词哈希对词汇表外问题具有鲁棒性，使我们能够将DNN解决方案扩展到需要极大词汇表的Web搜索任务中。我们将在第4节中展示该技术的优势。
>
> 在我们的实现中，基于字母n-gram的词哈希可以视为一个固定的（即非自适应的）线性变换，通过该变换，输入层中的术语向量被投影到更高层中的字母n-gram向量，如图1所示。由于字母n-gram向量的维度要低得多，因此可以有效地进行DNN学习。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/match/DSSM/Table1.png)

#### 3.3 Learning the DSSM

The clickthrough logs consist of a list of queries and their clicked documents. We assume that a query is relevant, at least partially, to the documents that are clicked on for that query. Inspired by the discriminative training approaches in speech and language processing , we thus propose a supervised training method to learn our model parameters, i.e., the weight matrices $W_i$ and bias vectors $b_i$ in our neural network as the essential part of the DSSM, so as to maximize the conditional likelihood of the clicked documents given the queries.

First, we compute the posterior probability of a document given a query from the semantic relevance score between them through a softmax function
$$
P(D \mid Q)=\frac{\exp (\gamma R(Q, D))}{\sum_{D^{\prime} \in D} \exp \left(\gamma R\left(Q, D^{\prime}\right)\right)}
$$
where $\gamma$ is a smoothing factor in the softmax function, which is set empirically on a held-out data set in our experiment. $\mathbf{D}$ denotes the set of candidate documents to be ranked. Ideally, $\mathbf{D}$ should contain all possible documents. In practice, for each (query, clicked-document) pair, denoted by where is a query and is the clicked document, we approximate $\mathbf{D}$ by including $D^+$ and four randomly selected unclicked documents, denote by $\left\{D_j^{-} ; j=1, \ldots, 4\right\}$. In our pilot study, we do not observe any significant difference when different sampling strategies were used to select the unclicked documents.

In training, the model parameters are estimated to maximize the likelihood of the clicked documents given the queries across the training set. Equivalently, we need to minimize the following loss function
$$
L(\Lambda)=-\log \prod_{\left(Q, D^{+}\right)} P\left(D^{+} \mid Q\right)
$$
where $\Lambda$ denotes the parameter set of the neural networks $\{W_i,b_i\}$ . Since $L(\Lambda)$ is differentiable w.r.t. to $\Lambda$, the model is trained readily using gradient-based numerical optimization algorithms. The detailed derivation is omitted due to the space limitation.

> 点击日志包含一系列查询及其被点击的文档。我们假设一个查询至少部分地与针对该查询被点击的文档相关。受到语音和语言处理中判别训练方法的启发，我们因此提出了一种监督训练方法来学习我们的模型参数，即作为DSSM基本部分的神经网络中的权重矩阵$W_i$和偏置向量$b_i$，以最大化给定查询下被点击文档的条件似然性。
>
> 首先，我们通过softmax函数从查询和文档之间的语义相关性得分中计算给定查询下文档的后验概率
>
> $$
> P(D \mid Q)=\frac{\exp (\gamma R(Q, D))}{\sum_{D^{\prime} \in D} \exp \left(\gamma R\left(Q, D^{\prime}\right)\right)}
> $$
>
> 其中，$\gamma$是softmax函数中的平滑因子，在我们的实验中，它根据一个保留的数据集经验性地设置。$\mathbf{D}$表示要排名的候选文档集。理想情况下，$\mathbf{D}$应该包含所有可能的文档。在实践中，对于每个（查询，被点击文档）对，表示为$(Q, D^+)$，其中$Q$是一个查询，$D^+$是被点击的文档，我们通过包括$D^+$和四个随机选择的未被点击的文档（表示为$\left\{D_j^{-} ; j=1, \ldots, 4\right\}$）来近似$\mathbf{D}$。在我们的初步研究中，当使用不同的抽样策略选择未被点击的文档时，我们没有观察到任何显著差异。
>
> 在训练中，模型参数被估计以最大化训练集中给定查询下被点击文档的似然性。等价地，我们需要最小化以下损失函数
>
> $$
> L(\Lambda)=-\log \prod_{\left(Q, D^{+}\right)} P\left(D^{+} \mid Q\right)
> $$
>
> 其中，$\Lambda$表示神经网络的参数集$\{W_i,b_i\}$。由于$L(\Lambda)$关于$\Lambda$是可微的，因此模型可以很容易地使用基于梯度的数值优化算法进行训练。由于篇幅限制，详细的推导过程在此省略。

#### 3.4 Implementation Details

To determine the training parameters and to avoid over-fitting, we divided the clickthrough data into two factions that do not overlap, called training and validation datasets, respectively. In our experiments, the models are trained on the training set and the training parameters are optimized on the validation dataset. For the DNN experiments, we used the architecture with three hidden layers as shown in Figure 1. The first hidden layer is the word hashing layer containing about 30k nodes (e.g., the size of the letter-trigrams as shown in Table 1). The next two hidden layers have 300 hidden nodes each, and the output layer has 128 nodes. Word hashing is based on a fixed projection matrix. The similarity measure is based on the output layer with the dimensionality of 128. Following [20], we initialize the network weights with uniform distribution in the range between $-\sqrt{6 /(\text { fanin }+ \text { fanout })}$ and $\sqrt{6 /(\text { fanin }+ \text { fanout })}$​ where and are the number of input and output units, respectively. Empirically, we have not observed better performance by doing layer-wise pre-training. In the training stage, we optimize the model using mini-batch based stochastic gradient descent (SGD). Each mini-batch consists of 1024 training samples. We observed that the DNN training usually converges within 20 epochs (passes) over the entire training data.

> 为了确定训练参数并避免过拟合，我们将点击数据分为两个不重叠的部分，分别称为训练集和验证集。在我们的实验中，模型在训练集上进行训练，并在验证集上优化训练参数。对于DNN实验，我们使用了具有三个隐藏层的架构，如图1所示。第一个隐藏层是包含约3万个节点的词哈希层（例如，如表1所示的字母三元组的大小）。接下来的两个隐藏层各有300个隐藏节点，输出层有128个节点。词哈希基于一个固定的投影矩阵。相似度度量基于维度为128的输出层。根据[20]，我们使用均匀分布在$-\sqrt{6 /(\text { fanin }+ \text { fanout })}$和$\sqrt{6 /(\text { fanin }+ \text { fanout })}$之间的值来初始化网络权重，其中$\text{fanin}$和$\text{fanout}$分别是输入和输出单元的数量。经验上，我们没有观察到通过逐层预训练可以获得更好的性能。在训练阶段，我们使用基于小批量的随机梯度下降（SGD）来优化模型。每个小批量包含1024个训练样本。我们观察到DNN训练通常在整个训练数据上经过20个周期（遍历）就能收敛。

## 4. EXPERIMENTS

We evaluated the DSSM, proposed in Section 3, on the Web document ranking task using a real-world data set. In this section, we first describe the data set on which the models are evaluated. Then, we compare the performances of our best model against other state of the art ranking models. We also investigate the break-down impact of the techniques proposed in Section 3.

> 我们使用真实世界的数据集在Web文档排名任务上评估了第3节中提出的DSSM。在本节中，我们首先描述用于评估模型的数据集。然后，我们将我们的最佳模型与其他最先进的排名模型进行比较。我们还研究了第3节中提出的技术的影响。

#### 4.1 Data Sets and Evaluation Methodology

We have evaluated the retrieval models on a large-scale real world data set, called the evaluation data set henceforth. The evaluation data set contains 16,510 English queries sampled from one-year query log files of a commercial search engine. On average, each query is associated with 15 Web documents (URLs). Each querytitle pair has a relevance label. The label is human generated and is on a 5-level relevance scale, 0 to 4, where level 4 means that the document is the most relevant to query $Q$ and 0 means $D$ is not relevant to $Q$ . All the queries and documents are preprocessed such that the text is white-space tokenized and lowercased, numbers are retained, and no stemming/inflection is performed.

All ranking models used in this study (i.e., DSSM, topic models, and linear projection models) contain many free hyperparameters that must be estimated empirically. In all experiments, we have used 2-fold cross validation: A set of results on one half of the data is obtained using the parameter settings optimized on the other half, and the global retrieval results are combined from the two sets.

The performance of all ranking models we have evaluated has been measured by mean Normalized Discounted Cumulative Gain (NDCG) [17], and we will report NDCG scores at truncation levels 1, 3, and 10 in this section. We have also performed a significance test using the paired t-test. Differences are considered statistically significant when the p-value is less than 0.05.

In our experiments, we assume that a query is parallel to the titles of the documents clicked on for that query. We extracted large amounts of the query-title pairs for model training from one year query log files using a procedure similar to [11]. Some previous studies, e.g., [24][11], showed that the query click field, when it is valid, is the most effective piece of information for Web search, seconded by the title field. However, click information is unavailable for many URLs, especially new URLs and tail URLs, leaving their click fields invalid (i.e., the field is either empty or unreliable because of sparseness). In this study, we assume that each document contained in the evaluation data set is either a new URL or a tail URL, thus has no click information (i.e., its click field is invalid). Our research goal is to investigate how to learn the latent semantic models from the popular URLs that have rich click information, and apply the models to improve the retrieval of those tail or new URLs. To this end, in our experiments only the title fields of the Web documents are used for ranking. For training latent semantic models, we use a randomly sampled subset of approximately 100 million pairs whose documents are popular and have rich click information. We then test trained models in ranking the documents in the evaluation data set containing no click information. The querytitle pairs are pre-processed in the same way as the evaluation data to ensure uniformity.

> 我们在一个大规模的真实世界数据集上评估了检索模型，该数据集被称为评估数据集。评估数据集包含从商业搜索引擎的一年查询日志文件中抽取的16,510个英文查询。平均而言，每个查询与15个Web文档（URL）相关联。每个查询-标题对都有一个相关性标签。该标签由人工生成，采用5级相关性量表，从0到4，其中4级表示文档与查询$Q$最相关，0级表示$D$与$Q$不相关。所有查询和文档都经过预处理，以便将文本进行空格分词并转换为小写，保留数字，不进行词干提取/词形变化。
>
> 本研究中使用的所有排名模型（即DSSM、主题模型和线性投影模型）都包含许多必须经验性估计的自由超参数。在所有实验中，我们都使用了2折交叉验证：一半数据的结果是使用在另一半数据上优化的参数设置获得的，全局检索结果来自两组结果的组合。
>
> 我们评估的所有排名模型的性能都通过平均归一化折扣累积增益（NDCG）来测量[17]，并且在本节中将报告截断级别为1、3和10的NDCG得分。我们还使用了配对t检验进行了显著性检验。当p值小于0.05时，差异被视为具有统计显著性。
>
> 在我们的实验中，我们假设一个查询与针对该查询被点击的文档的标题是平行的。我们使用类似于[11]的过程从一年的查询日志文件中提取了大量用于模型训练的查询-标题对。一些先前的研究，例如[24][11]，表明当查询点击字段有效时，它是Web搜索中最有效的信息，其次是标题字段。然而，许多URL，尤其是新URL和长尾URL，没有点击信息，使得它们的点击字段无效（即该字段为空或由于稀疏性而不可靠）。在本研究中，我们假设评估数据集中包含的每个文档都是新URL或长尾URL，因此没有点击信息（即其点击字段无效）。我们的研究目标是调查如何从具有丰富点击信息的流行URL中学习潜在语义模型，并将这些模型应用于改进那些长尾或新URL的检索。为此，在我们的实验中仅使用Web文档的标题字段进行排名。为了训练潜在语义模型，我们使用了一个随机抽样的子集，大约包含1亿对文档，这些文档很受欢迎且具有丰富的点击信息。然后，我们在包含没有点击信息的文档的评估数据集中测试训练后的模型进行排名。查询-标题对与评估数据以相同的方式进行预处理，以确保一致性。

#### 4.2 Results

The main results of our experiments are summarized in Table 2, where we compared our best version of the DSSM (Row 12) with three sets of baseline models. The first set of baselines includes a couple of widely used lexical matching methods such as TF-IDF (Row 1) and BM25 (Row 2). The second is a word translation model (WTM in Row 3) which is intended to directly address the query-document language discrepancy problem by learning a lexical mapping between query words and document words [9][10]. The third includes a set of state-of-the-art latent semantic models which are learned either on documents only in an unsupervised manner (LSA, PLSA, DAE as in Rows 4 to 6) or on clickthrough data in a supervised way (BLTM-PR, DPM, as in Rows 7 and 8). In order to make the results comparable, we reimplement these models following the descriptions in [10], e.g., models of LSA and DPM are trained using a 40k-word vocabulary due to the model complexity constraint, and the other models are trained using a 500K-word vocabulary. Details are elaborated in the following paragraphs.

**TF-IDF** (Row 1) is the baseline model, where both documents and queries represented as term vectors with TF-IDF term weighting. The documents are ranked by the cosine similarity between the query and document vectors. We also use BM25 (Row 2) ranking model as one of our baselines. Both TF-IDF and BM25 are state-of-the-art document ranking models based on term matching. They have been widely used as baselines in related studies.

**WTM** (Rows 3) is our implementation of the word translation model described in [9], listed here for comparison. We see that WTM outperforms both baselines (TF-IDF and BM25) significantly, confirming the conclusion reached in [9]. LSA (Row 4) is our implementation of latent semantic analysis model. We used PCA instead of SVD to compute the linear projection matrix. Queries and titles are treated as separate documents, the pair information from the clickthrough data was not used in this model. PLSA (Rows 5) is our implementation of the model proposed in [15], and was trained on documents only (i.e., the title side of the query-title pairs). Different from [15], our version of PLSA was learned using MAP estimation as in [10]. DAE (Row 6) is our implementation of the deep auto-encoder based semantic hashing model proposed by Salakhutdinov and Hinton in [22]. Due to the model training complexity, the input term vector is based on a 40k-word vocabulary. The DAE architecture contains four hidden layers, each of which has 300 nodes, and a bottleneck layer in the middle which has 128 nodes. The model is trained on documents only in an unsupervised manner. In the fine-tuning stage, we used cross-entropy error as training criteria. The central layer activations are used as features for the computation of cosine similarity between query and document. Our results are consistent with previous results reported in [22]. The DNN based latent semantic model outperforms the linear projection model (e.g., LSA). However, both LSA and DAE are trained in an unsupervised fashion on document collection only, thus cannot outperform the state-of-the-art lexical matching ranking models. 

**BLTM-PR** (Rows 7) is the best performer among different versions of the bilingual topic models described in [10]. BLTM with posterior regularization (BLTM-PR) is trained on query-title pairs using the EM algorithm with a constraint enforcing the paired query and title to have same fractions of terms assigned to each hidden topic. DPM (Row 8) is the linear discriminative projection model proposed in [10], where the projection matrix is discriminatively learned using the S2Net algorithm [26] on relevant and irrelevant pairs of queries and titles. Similar to that BLTM is an extension to PLSA, DPM can also be viewed as an extension of LSA, where the linear projection matrix is learned in a supervised manner using clickthrough data, optimized for document ranking. We see that using clickthrough data for model training leads to some significant improvement. Both BLTM-PR and DPM outperform the baseline models (TF-IDF and BM25)

> 我们实验的主要结果总结在表2中，其中我们将最佳版本的DSSM（第12行）与三组基线模型进行了比较。第一组基线模型包括几种广泛使用的基于词汇的匹配方法，如TF-IDF（第1行）和BM25（第2行）。第二组是一个单词翻译模型（第3行的WTM），旨在通过学习查询词和文档词之间的词汇映射来直接解决查询-文档语言差异问题[9][10]。第三组包括一组最先进的潜在语义模型，这些模型要么仅以无监督方式在文档上学习（如第4至6行的LSA、PLSA、DAE），要么以监督方式在点击数据中学习（如第7和8行的BLTM-PR、DPM）。为了使结果具有可比性，我们根据[10]中的描述重新实现了这些模型，例如，由于模型复杂性限制，LSA和DPM模型使用4万个词的词汇表进行训练，而其他模型使用50万个词的词汇表进行训练。以下段落将详细阐述这些模型的细节。
>
> **TF-IDF**（第1行）是基线模型，其中文档和查询均表示为具有TF-IDF术语权重的术语向量。文档根据查询和文档向量之间的余弦相似度进行排名。我们还使用BM25（第2行）排名模型作为我们的基线之一。TF-IDF和BM25都是基于术语匹配的先进文档排名模型，已被广泛用作相关研究的基线。
>
> **WTM**（第3行）是我们在[9]中描述的单词翻译模型的实现，列在这里进行比较。我们看到WTM显著优于两个基线（TF-IDF和BM25），这证实了[9]中得出的结论。LSA（第4行）是我们实现的潜在语义分析模型。我们使用PCA代替SVD来计算线性投影矩阵。查询和标题被视为单独的文档，点击数据中的配对信息未在该模型中使用。PLSA（第5行）是我们实现的[15]中提出的模型，并且仅使用文档（即查询-标题对的标题侧）进行训练。与[15]不同，我们的PLSA版本使用MAP估计进行学习，如[10]所述。DAE（第6行）是我们实现的基于深度自动编码器的语义哈希模型，由Salakhutdinov和Hinton在[22]中提出。由于模型训练复杂性，输入术语向量基于4万个词的词汇表。DAE架构包含四个隐藏层，每层有300个节点，中间有一个128个节点的瓶颈层。该模型仅以无监督方式在文档上进行训练。在微调阶段，我们使用交叉熵误差作为训练标准。中间层的激活用作计算查询和文档之间余弦相似性的特征。我们的结果与[22]中报告的结果一致。基于DNN的潜在语义模型优于线性投影模型（例如LSA）。然而，LSA和DAE均仅以无监督方式在文档集合上进行训练，因此无法超越最先进的基于词汇匹配的排名模型。
>
> **BLTM-PR**（第7行）是在[10]中描述的不同版本双语主题模型中的最佳表现者。具有后验正则化的BLTM（BLTM-PR）使用EM算法在查询-标题对上进行训练，同时通过一个约束强制成对的查询和标题具有相同的术语分配给每个隐藏主题的分数。DPM（第8行）是在[10]中提出的线性判别投影模型，其中投影矩阵使用S2Net算法[26]在相关和不相关查询-标题对上以判别方式学习。与BLTM是PLSA的扩展类似，DPM也可以视为LSA的扩展，其中线性投影矩阵使用点击数据以监督方式学习，针对文档排名进行优化。我们看到，使用点击数据进行模型训练带来了一些显著的改进。BLTM-PR和DPM均优于基线模型（TF-IDF和BM25）。

Rows 9 to 12 present results of different settings of the DSSM. DNN (Row 9) is a DSSM without using word hashing. It uses the same structure as DAE (Row 6), but is trained in a supervised fashion on the clickthrough data. The input term vector is based on a 40k-word vocabulary, as used by DAE. L-WH linear (Row 10) is the model built using letter trigram based word hashing and supervised training. It differs from the L-WH nonlinear model (Row 11) in that we do not apply any nonlinear activation function, such as tanh, to its output layer. L-WH DNN (Row 12) is our best DNN-based semantic model, which uses three hidden layers, including the layer with the Letter-trigrambased Word Hashing (L-WH), and an output layer, and is discriminatively trained on query-title pairs, as described in Section 3. Although the letter n-gram based word hashing method can be applied to arbitrarily large vocabularies, in order to perform a fair comparison with other competing methods, the model uses a 500K-word vocabulary. 

The results in Table 2 show that the deep structured semantic model is the best performer, beating other methods by a statistically significant margin in NDCG and demonstrating the empirical effectiveness of using DNNs for semantic matching.

From the results in Table 2, it is also clear that supervised learning on clickthrough data, coupled with an IR-centric optimization criterion tailoring to ranking, is essential for obtaining superior document ranking performance. For example, both DNN and DAE (Row 9 and 6) use the same 40k-word vocabulary and adopt the same deep architecture. The former outperforms the latter by 3.2 points in NDCG@1.

Word hashing allows us to use very large vocabularies for modeling. For instance, the models in Rows 12, which use a 500kword vocabulary (with word hashing), significantly outperform the model in Row 9, which uses a 40k-word vocabulary, although the former has slightly fewer free parameters than the later since the word hashing layer containing about only 30k nodes. 

We also evaluated the impact of using a deep architecture versus a shallow one in modeling semantic information embedded in a query and a document. Results in Table 2 show that DAE (Row 3) is better than LSA (Row 2), while both LSA and DAE are unsupervised models. We also have observed similar results when comparing the shallow vs. deep architecture in the case of supervised models. Comparing models in Rows 11 and 12 respectively, we observe that increasing the number of nonlinear layers from one to three raises the NDCG scores by 0.4-0.5 point which are statistically significant, while there is no significant difference between linear and non-linear models if both are onelayer shallow models (Row 10 vs. Row 11).

> 表2的第9至12行展示了DSSM不同设置的结果。DNN（第9行）是一个不使用词哈希的DSSM。它与DAE（第6行）具有相同的结构，但使用点击数据以监督方式进行训练。输入术语向量基于与DAE相同的4万个词的词汇表。L-WH linear（第10行）是使用基于字母三元组的词哈希和监督训练构建的模型。它与L-WH nonlinear模型（第11行）的不同之处在于，我们没有对其输出层应用任何非线性激活函数，如tanh。L-WH DNN（第12行）是我们最佳的基于DNN的语义模型，它使用三个隐藏层，包括基于字母三元组的词哈希层（L-WH）和输出层，并在查询-标题对上以判别方式进行训练，如第3节所述。虽然基于字母n-gram的词哈希方法可以应用于任意大的词汇表，但为了与其他竞争方法进行公平比较，该模型使用了50万个词的词汇表。
>
> 表2的结果显示，深度结构语义模型表现最佳，在NDCG方面以统计上显著的优势击败了其他方法，证明了使用DNN进行语义匹配的实证有效性。
>
> 从表2的结果还可以清楚地看出，在点击数据上进行监督学习，并结合针对排名的以IR为中心的优化标准，对于获得优异的文档排名性能至关重要。例如，DNN和DAE（第9行和第6行）都使用相同的4万个词的词汇表，并采用相同的深度架构。前者在NDCG@1上比后者高出3.2分。
>
> 词哈希使我们能够使用非常大的词汇表进行建模。例如，第12行的模型使用了一个50万个词的词汇表（带有词哈希），显著优于使用4万个词的词汇表的第9行模型，尽管前者包含的自由参数略少于后者，因为词哈希层仅包含约3万个节点。
>
> 我们还评估了在建模查询和文档中嵌入的语义信息时使用深度架构与浅层架构的影响。表2的结果显示，DAE（第6行）优于LSA（第4行），而LSA和DAE都是无监督模型。在监督模型的情况下，我们也观察到了类似的结果。比较第11行和第12行的模型，我们发现将非线性层的数量从一层增加到三层可以使NDCG得分提高0.4-0.5分，这在统计上是显著的，但如果两者都是一层浅层模型，则线性和非线性模型之间没有显著差异（第10行与第11行）。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/match/DSSM/Table2.png)

## 5. CONCLUSIONS

We present and evaluate a series of new latent semantic models, notably those with deep architectures which we call the DSSM. The main contribution lies in our significant extension of the previous latent semantic models (e.g., LSA) in three key aspects. First, we make use of the clickthrough data to optimize the parameters of all versions of the models by directly targeting the goal of document ranking. Second, inspired by the deep learning framework recently shown to be highly successful in speech recognition [5][13][14][16][18], we extend the linear semantic models to their nonlinear counterparts using multiple hiddenrepresentation layers as. The deep architectures adopted have further enhanced the modeling capacity so that more sophisticated semantic structures in queries and documents can be captured and represented. Third, we use a letter n-gram based word hashing technique that proves instrumental in scaling up the training of the deep models so that very large vocabularies can be used in realistic web search. In our experiments, we show that the new techniques pertaining to each of the above three aspects lead to significant performance improvement on the document ranking task. A combination of all three sets of new techniques has led to a new state-of-the-art semantic model that beats all the previously developed competing models with a significant margin.

> 我们介绍并评估了一系列新的潜在语义模型，特别是那些具有深度架构的我们称之为DSSM的模型。我们的主要贡献在于在三个关键方面对先前的潜在语义模型（例如LSA）进行了显著扩展。
>
> 首先，我们利用点击数据直接针对文档排名的目标来优化所有版本模型的参数。
>
> 其次，受最近在语音识别中表现出高度成功的深度学习框架的启发[5][13][14][16][18]，我们使用多个隐藏表示层将线性语义模型扩展为其非线性对应物。所采用的深度架构进一步增强了建模能力，从而可以捕获和表示查询和文档中的更复杂的语义结构。
>
> 第三，我们使用了一种基于字母n-gram的词哈希技术，该技术对于扩大深度模型的训练规模至关重要，以便在现实的Web搜索中使用非常大的词汇表。在我们的实验中，我们展示了与上述三个方面相关的新技术都导致了文档排名任务的显著性能提升。结合所有三组新技术的组合，我们得到了一种新的最先进的语义模型，它以显著的优势击败了所有先前开发的竞争模型。
