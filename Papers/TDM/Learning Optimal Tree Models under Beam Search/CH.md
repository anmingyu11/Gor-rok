# Learning Optimal Tree Models under Beam Search

## Abstract

在计算资源受限的情况下从一个极大的目标数据集合中检索相关目标是信息检索和推荐系统面临的共同挑战。树模型将目标表达为可训练的非叶子结点和叶子结点组成的评分器，由于其在训练和测试中的对数计算复杂度，因此吸引了很多关注来应对这一挑战。TDMs 和 PLTs 是其中两种典型的模型。尽管取得了许多实际上的成功，但现有的树模型存在训练-测试差异的问题，在训练过程中未考虑测试中由于 beam search 导致的检索能力下降。这导致了最相关目标与通过 beam search 检索到的目标之间存在固有差距，即使使用最优的节点评分器也无法消除此差距。我们首次迈出了理论上理解和分析这个问题的一步，并提出了 beam search 下的贝叶斯最优性和 beam search 的校准概念作为通用的分析工具。此外，为了消除这种差异，我们提出了一种学习 beam search 下最优树模型的新算法。在合成数据和真实数据上的实验证实了我们理论分析的合理性，并证明了我们的算法相比最先进的方法的优越性。

## 1. Introduction

在现代信息检索和推荐系统的工业应用中，广泛存在超大规模检索问题。例如，在在线广告系统中，需要从一个包含数千万个广告的目标集合中检索出多个广告，并在几十毫秒内展示给用户。计算资源和响应时间的限制使得那些计算复杂度与目标集合大小成线性关系的模型在实践中变得不可接受。

树模型对于解决这些问题特别有意义，因为它们在训练和测试中都可以达到对数复杂度。基于树结构的深度模型（TDMs）和概率标签树（PLTs）是两种有代表性的树模型。这些模型引入了一个树层次结构，其中每个叶节点对应一个目标，每个非叶节点定义了一个伪目标，用于衡量其子树上存在相关目标的程度。每个节点还与一个节点评分器相关联，该评分器被训练以估计相应（伪）目标的相关概率。为了实现对数级的训练复杂度，使用下采样方法选择要对每个训练实例选择对数个数的节点来训练评分器。在测试中，通常使用 beam search 来以对数复杂度检索相关目标。

作为一种贪心法，beam search 仅展开得分较高的部分节点，同时剪枝其他节点。这种性质确保了对数级的计算复杂度，但如果最相关目标的祖先节点被修剪，则可能导致检索效果变差。理想的树模型应该在使用节点评分器进行 beam search 时不会导致效果下降。然而，现有的树模型忽略了这一点，并将训练视为与测试分离的任务：

1. 节点评分器被训练为估计非最优检索的伪目标的概率；
2. 它们还在与测试中的 beam search 不同的子采样节点上进行训练。这种差异使得即使针对训练损失而言最优的节点评分器在测试中通过 beam search 检索相关目标时也可能导致次优的结果。据我们所知，目前很少有基于理论或实验证明这个问题的工作。

我们迈出了解决树模型的训练和测试差异的第一步。为了形式化地分析这个问题，我们引入了 beam search 下的贝叶斯最优性和校准作为树模型和相应训练损失的最优度量。这两个概念都可以作为树模型的通用分析工具。基于这些概念，我们证明了 TDMs 和 PLTs 都不是最优的，并推导出最优树模型存在的充分条件。我们还提出了一种新的算法来学习这样一个最优的树模型。我们的算法包括一个考虑 beam search 的下采样方法和一种基于最优检索的伪目标定义，这两种方法解决了训练和测试之间的差异。在合成数据和真实数据上的实验不仅验证了我们新提出的概念在衡量树模型最优性方面的合理性，还证明了我们的算法相对于现有最先进方法的优越性。

## 2. Related Work

**树模型**：对树模型的研究主要集中在节点评分器和树结构的建模上。对于节点评分器，广泛采用线性模型，而近年来深度模型变得流行起来。对于树结构，除了随机树，最近的研究提出通过层次聚类目标或在节点评分器的联合优化框架下学习树结构。不依赖于特定的节点评分器或树结构的具体形式，我们的理论发现和提出的训练算法是通用的，并适用于这些进展。

**贝叶斯最优性和校准**：贝叶斯最优性和校准在一般模型上得到了广泛的研究（，并且它们也被用于衡量树模型在层次概率估计上的性能（Wydmuch等，2018）。然而，在层次概率估计和相关目标检索之间存在差距，因为前者忽略了 beam search 和相应的性能恶化。因此，如何形式上衡量树模型的目标检索性能仍然是一个开放的问题。我们通过在 beam search 下发展贝叶斯最优性和校准的概念来填补这一空白。

**训练中的 beam search**：将 beam search 纳入训练以解决训练-测试差异并不是一个新思路。它已经在结构化预测模型（如机器翻译和语音识别）上进行了广泛的研究。尽管通过 beam search 导致的性能恶化已经经验性地进行了分析，但仍然缺乏理论上的理解。此外，在理解和解决树模型上的训练-测试差异方面几乎没有任何进展。我们首先从理论和实验两方面对这些问题进行研究。

## 3. Preliminaries

### 3.1. Problem Definition

假设 $\mathcal{I} = \{1, ..., M\}$，其中 $M \gg 1$ 是目标集合，$\mathcal{X}$ 是观测空间。我们将一个实例表示为 $(\mathbf{x}, \mathcal{I}_\mathbf{x})$，表示观测 $\mathbf{x} \in \mathcal{X}$ 与相关目标的子集 $\mathcal{I}_\mathbf{x} \subset \mathcal{I}$ 相关联，通常满足 $|\mathcal{I}_\mathbf{x}| \ll M$。为了简化表示，我们引入二进制向量$\mathbf{y} \in \mathcal{Y} = \{0, 1\}^M$ 作为 $\mathcal{I}_x$ 的替代表示，其中 $y_j = 1$ 意味着 $j \in \mathcal{I}_\mathbf{x}$，反之亦然。因此，一个实例也可以表示为 $(\mathbf{x}, \mathbf{y}) \in \mathcal{X} \times \mathcal{Y}$。

设 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 是一个在实践中未知的数据概率密度函数，我们稍微滥用符号，将实例$(\mathbf{x}, \mathbf{y})$视为关于$p(\mathbf{x}, \mathbf{y})$的随机变量对或$p(\mathbf{x}, \mathbf{y})$的一个样本。我们还假设训练数据集$\mathcal{D}_{tr}$和测试数据集$\mathcal{D}_{te}$是包含$p(\mathbf{x}, \mathbf{y})$的独立同分布样本的集合。由于$\mathbf{y}$是一个二进制向量，在本文的其余部分中，我们使用简化的记法 $\eta_j(\mathbf{x})=p(y_j=1 \mid \mathbf{x})$ 来表示任意 $j \in \mathcal{I}$。

在这些符号的基础上，极大规模的检索问题被定义为学习一个模型$\mathcal{M}$，使得对于任意$\mathbf{x} \sim p(\mathbf{x})$，其检索到的子集，用 $\hat{\mathcal{I}}_{\mathbf{x}}$ 或 $\hat{\mathbf{y}}$ 表示，根据某些性能度量指标与 $\mathbf{y} \sim p(\mathbf{y} \mid \mathbf{x})$ 尽可能接近。由于实际中 $p(\mathbf{x}, \mathbf{y})$ 是未知的，因此这样的模型通常作为 $p(\mathbf{y} \mid \mathbf{x})$ 的估计器在 $D_{tr}$ 上进行学习，并在 $D_{te}$ 上评估其检索性能。

### 3.2. Tree Models

假设 $\mathcal{T}$ 是一个高度为 $H$ 的 $b$ 元树，我们将第 $0$ 层的节点视为根节点，第 $H$ 层的节点视为叶子节点。形式上，我们将第 $h$ 层的节点集合表示为 $\mathcal{N}_h$，将 $\mathcal{T}$ 的节点集合表示为 $\mathcal{N}=\bigcup_{h=0}^H \mathcal{N}_h$。对于每个节点$n \in \mathcal{N}$，我们用 $\rho(n) \in \mathcal{N}$ 表示其父节点，用 $\mathcal{C}(n) \subset \mathcal{N}$ 表示其子节点集合，用 $Path(n)$ 表示从根节点到 $n$ 的路径，用 $\mathcal{L}(n)$ 表示其子树上的叶子节点集合。

通过一个双射映射 $π : \mathcal{N}_H → \mathcal{I}$，树模型将目标集合 $\mathcal{I}$ 表示为 $\mathcal{T}$ 的叶子节点，这意味着$H = O(log_bM)$。对于任意实例 $(\mathbf{x}, \mathbf{y})$，每个节点 $n ∈ \mathcal{N}$ 都定义了一个伪目标 $z_n ∈ \{0, 1\}$ 来衡量在 $n$ 的子树上是否存在相关目标，即， 
$$
z_n=\mathbb{I}\left(\sum_{n^{\prime} \in \mathcal{L}(n)} y_{\pi\left(n^{\prime}\right)} \geq 1\right)
\\(1)
$$
其中对于 $n ∈ \mathcal{N}_H$，满足 $z_n = y_{π(n)}$ 。

通过这样做，树模型将原始的估计问题 $p(y_j |\mathbf{x})$ 转化为一系列层次子问题，即在 $n ∈ Path(π^{−1}(j))$ 上估计 $p(z_n|\mathbf{x})$。它们引入了逐节点评分函数 $g : \mathcal{X} × \mathcal{N} → \mathbb{R}$ 来为每个 $n ∈ \mathcal{N}$ 构建一个逐节点的估计器，该估计器用 $p_g(z_n|\mathbf{x})$ 表示，以区别于未知的分布 $p(z_n|\mathbf{x})$。在本文的其余部分，我们将树模型表示为 $M(\mathcal{T} , g)$，以强调其对 $\mathcal{T}$ 和 $g$ 的依赖关系。

#### 3.2.1. TRAINING OF TREE MODELS

树模型的训练损失可以表示为 $\operatorname{argmin}_g \sum_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}_{t r}} L(\mathbf{y}, \mathbf{g}(\mathbf{x}))$，其中 
$$
L(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \mathcal{S}_h(\mathbf{y})} \ell_{\mathrm{BCE}}\left(z_n, g(\mathbf{x}, n)\right)
\\(2)
$$
在公式 (2) 中，$\mathbf{g}(\mathbf{x})$ 是 $\{g(\mathbf{x}, n) : n ∈ \mathcal{N} \}$ 的向量化表示（例如，按层次遍历），$l_{BCE}(z, g) = −z log(1 + exp(−g)) − (1 − z) log(1 + exp(g))$ 是二分类的交叉熵损失函数，$\mathcal{S}_h(y) ⊂ \mathcal{N}_h$ 是实例 $(\mathbf{x}, \mathbf{y})$ 在第 $h$ 层的下采样节点集合。令 $C = max_h |\mathcal{S}_h(y)|$，训练复杂度是每个实例的 $O(HbC)$，这对于目标集大小 $M$ 是对数级别的。 作为树模型的两个代表，PLTs 和 TDMs 采用不同的方式构建 $p_g$ 和 $S_h(y)$。

**PLTs**（Path Learning Trees）: 根据公式 (1)，$p\left(z_n \mid \mathbf{x}\right)$ 可以分解为 $p\left(z_n=1 \mid \mathbf{x}\right)= \prod_{n^{\prime} \in \operatorname{Path}(n)} p\left(z_{n^{\prime}}=1 \mid z_{\rho\left(n^{\prime}\right)}=1, \mathbf{x}\right)$，因此，通过以下方式对 $p_g\left(z_n \mid \mathbf{x}\right)$ 进行相应的分解：
$$
p_g\left(z_{n^{\prime}} \mid z_{\rho\left(n^{\prime}\right)}=\right. 1, \mathbf{x})=1 /\left(1+\exp \left(-\left(2 z_{n^{\prime}}-1\right) g\left(\mathbf{x}, n^{\prime}\right)\right)\right)
$$
结果是只有当 $z_{\rho(n)}=1$ 时，才会进行训练，从而产生 $\mathcal{S}_h(\mathbf{y})=\left\{n: z_{\rho(n)}=1, n \in \mathcal{N}_h\right\}$。 

**TDMs**（Tree Dropout Models）: 与 PLTs 不同，通过 $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$ 直接估计 $p(z_n|\mathbf{x})$。此外，子采样集合选择为 $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \bigcup \mathcal{S}_h^{-}(\mathbf{y})$，其中 $\mathcal{S}_h^{+}(\mathbf{y})=\left\{n: z_n=1, n \in \mathcal{N}_h\right\}$，而 $\mathcal{S}_h^{-}(\mathbf{y})$ 包含从 $\mathcal{N}_h \backslash \mathcal{S}_h^{+}(\mathbf{y})$ 中随机抽取的若干样本。

#### 3.2.2. TESTING OF TREE MODELS

对于任意的测试实例 $(\mathbf{x}, \mathbf{y})$，假设 $\mathcal{B}_h(\mathbf{x})$ 表示通过 Beam Search 检索得到的第 $h$ 层的节点集合，并且 $k=\left|\mathcal{B}_h(\mathbf{x})\right|$ 表示 Beam 大小（即保留的节点数），则 Beam Search 过程可以定义为： 
$$
\mathcal{B}_h(\mathbf{x}) \in \underset{n \in \tilde{\mathcal{B}}_h(\mathbf{x})}{\arg \operatorname{Topk}} p_g\left(z_n=1 \mid \mathbf{x}\right)
\\(3)
$$
其中 $\tilde{\mathcal{B}}_h(\mathbf{x})=\bigcup_{n^{\prime} \in \mathcal{B}_{h-1}(\mathbf{x})} \mathcal{C}\left(n^{\prime}\right)$，表示通过将上一层的节点 $n^{\prime}$ 映射到其对应的子节点集合 $\mathcal{C}\left(n^{\prime}\right)$，构建当前层的候选节点集合。然后，从这个候选节点集合中选择概率 $p_g\left(z_n=1 \mid \mathbf{x}\right)$ 最高的前 $k$ 个节点作为当前层的节点集合 $\mathcal{B}_h(\mathbf{x})$。

通过将公式 (3) 递归应用直到 $h = H$，Beam Search 将检索出包含 $k$ 个叶节点的集合，记为 $\mathcal{B}_H(\mathbf{x})$。假设 $m \leq k$ 表示要检索的目标数量，则检索得到的目标子集可以表示为 
$$
\hat{\mathcal{I}}_{\mathbf{x}}=\left\{\pi(n): n \in \mathcal{B}_H^{(m)}(\mathbf{x})\right\} \\ (4)
$$
其中 $\mathcal{B}_H^{(m)}(\mathbf{x})$ 表示从 $\mathcal{B}_H(\mathbf{x})$ 中选择概率最高的 $m$ 个节点，并通过函数 $\pi(\cdot)$ 将这些节点映射到对应的目标。

其中 $\mathcal{B}_H^{(m)}(\mathbf{x}) \in \operatorname{argTopm}_{n \in \mathcal{B}_H(\mathbf{x})} p_g\left(z_n=1 \mid \mathbf{x}\right)$ 表示根据 $p_g\left(z_n=1 \mid \mathbf{x}\right)$ 得分，从 $\mathcal{B}_H(\mathbf{x})$ 中选择得分最高的 $m$ 个节点构成的子集。由于公式 (3) 最多遍历 $b k$ 个节点，并且生成 $\mathcal{B}_H(\mathbf{x})$ 需要计算公式 (3) $H$ 次，每个实例的测试复杂度为 $\mathrm{O}(H b k)$，这也是对 $M$ 取对数的。 为了评估 $\mathcal{M}(\mathcal{T}, g)$ 在测试数据集 $D_{te}$ 上的检索性能，通常使用 Precision@m、Recall@m 和 F-measure@m 进行评估。我们将它们定义为分别在 $D_{te}$ 上对公式 (5)、公式 (6) 和公式 (7) 求平均值，其中 


$$
\mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{m} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j \\(5)
$$

$$
\mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{\left|\mathcal{I}_{\mathbf{x}}\right|} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j \\(6)
$$

$$
\mathrm{F} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{2 \cdot \mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y}) \cdot \mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})}{\mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})+\mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})} \\(7)
$$

> 译注：把以前的TDM工作给形式化了一遍。

## 4. Main Contributions

我们的主要贡献可以分为三个部分：

1. 我们强调了树模型中存在的训练-测试差异，并对其对检索性能的负面影响提供了直观的解释；
2. 我们提出了在 Beam Search 下的贝叶斯最优性和在 Beam Search 下的校准的概念，以形式化这种直观解释；
3. 我们提出了一种新颖的算法，用于学习在 Beam Search 下是贝叶斯最优的树模型。

### 4.1. Understanding the Training-Testing Discrepancy on Tree Models

根据公式 (2)，$g(\mathbf{x}, n)$ 的训练取决于两个因素：子样本集 $\mathcal{S}_h(\mathbf{y})$ 和伪目标 $z_n$。

我们可以证明这两个因素与现有树模型中的训练-测试差异有关。

首先，根据公式 (3)，在测试集中查询 $g(\mathbf{x}, n)$ 的第 $h$ 层节点可以表示为 $\tilde{\mathcal{B}}_h(\mathbf{x})$，这意味着 $g(\mathbf{x}, n)$ 存在自依赖性，即在第 $h$ 层查询 $g(\mathbf{x}, n)$ 的节点取决于在第 $(h − 1)$ 层查询 $g(\mathbf{x}, n)$ 的节点。然而，$\mathcal{S}_h(\mathbf{y})$，即训练 $g(\mathbf{x}, n)$ 的第 $h$ 层节点，是通过公式 (1) 根据真实目标 $\mathbf{y}$ 生成的。图 1(a) 和图 1(b) 展示了这种差异：节点 7 和 8（蓝色节点）由 Beam Search 遍历，但它们不在 PLT 的 $\mathcal{S}_h(\mathbf{y})$ 中，也可能不在 TDM 的 $\mathcal{S}_h(\mathbf{y})$ 中，根据 $\mathcal{S}_h^{+}(\mathbf{y})$（红色节点）。因此，当使用 $g(\mathbf{x}, n)$ 通过 Beam Search 检索相关目标时，它的训练并没有考虑到这种自依赖性。这种差异导致了即使 $g(\mathbf{x}, n)$ 训练得很好，在测试中表现也不好。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Figure1.png)

Figure 1. An overview of the training-testing discrepancy on a tree model $\mathcal{M}(\mathcal{T}, g)$. (a) The assignment of pseudo targets on existing tree models, where red nodes correspond to  $z_n=1$ defined in Eq. (1). (b) Beam search process, where targets mapping blue nodes at 3-th level (i.e., leaf nodes) are regarded as the retrieval results of $\mathcal{M}$. (c) The assignment of optimal pseudo targets based on the ground truth distribution $\eta_j(\mathbf{x})=p\left(y_j=1 \mid \mathbf{x}\right)$, where green nodes correspond to $z_n^*=1$ defined in Eq. (13).



![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table1.png)

Table 1. Results for the toy experiment with $M = 1000$, $b = 2$. The reported number is $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$ which is averaged over 100 runs with random initialization over $\mathcal{T}$ and $η_j$ .

第二，根据公式（1）定义的 $z_n$ 不能保证针对 $p_g(z_n = 1|\mathbf{x})$ 的 beam search 不会出现性能下降，即无法检索到最相关的目标。为了看清这一点，我们设计了一个简单的例子，忽略 $\mathbf{x}$ 并将数据分布定义为 $p(\mathbf{y})=\prod_{j=1}^M p\left(y_j\right)$，其中边缘概率 $\eta_j=p\left(y_j=1\right)$ 是从 $[0, 1]$ 的均匀分布中采样得到的。因此，我们将训练数据集表示为 $\mathcal{D}_{tr}=\left\{\mathbf{y}^{(i)}\right\}_{i=1}^N$，对于节点 $n$ 上的实例 $\mathbf{y}^{(i)}$，伪目标被表示为 $z^{(i)}_n$。对于$\mathcal{M}(\mathcal{T}, g)$，我们假设 $\mathcal{T}$ 是随机构建的，并通过 $p_g\left(z_n=1\right)=\sum_{i=1}^N z_n^{(i)} / N$ 直接估计 $p\left(z_n=1\right)$，而无需指定 $g$，因为没有观察到 $\mathbf{x}$。在 $\mathcal{M}$ 上应用大小为 $k$ 的beam sarch，以检索目标子集，其大小也为$m=k$，表示为$\hat{\mathcal{I}}=\left\{\pi(n): n \in \mathcal{B}_H\right\}$。由于在这个简单的例子中 $p(\mathbf{y})$ 是已知的，我们不需要测试集 $\mathcal{D}_{te}$，而是直接通过 $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$ 来评估检索性能，其中 $\mathcal{I}^{(k)} \in \operatorname{argTopk}_{j \in \mathcal{I}} \eta_j$ 表示根据 $η_j$ 获取前 $k$ 个目标的集合。作为公式（10）的一种特殊情况，该度量指标衡量了 $\mathcal{M}$ 的次优性，我们将在后面正式讨论它。

> 译注：不止预测正样本，对 beam search的topk 也预测。

### 4.2. Bayes Optimality and Calibration under Beam Search

在4.1节中，我们讨论了树模型中训练和测试之间差异的存在，并提供了一个简单的例子来解释其影响。为了方便起见，我们在本小节中将此讨论形式化为以Precision@m为检索性能度量。 

首先要解决的问题是，对于树模型而言，与其检索性能相关的“最优”是什么意思。实际上，在4.1节的例子中部分揭示了答案，我们给出以下正式定义： 

**定义1**（beam search 下的贝叶斯最优性）。给定 beam 大小 $k$ 和数据分布 $p: \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$，如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 都有
$$
\left\{\pi(n): n \in \mathcal{B}_H(\mathbf{x})\right\} \in \underset{j \in \mathcal{I}}{\operatorname{argTopk}} \eta_j(\mathbf{x}) \\ (8)
$$
则称树模型 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是 top-$k$ 贝叶斯最优的。如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 和 $1 ≤ k ≤ M$ 都满足公式（8），则称 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是贝叶斯最优的。

根据定义1，我们可以推导出一个充分条件，来确保存在这样一个最优的树模型，如下所示： 定义1（ beam search 下贝叶斯最优性的充分条件）。给定 beam 大小 $k$，数据分布 $p : \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$ ，树 $\mathcal{T}$ 和 
$$
p^*\left(z_n \mid \mathbf{x}\right)= \begin{cases}\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=1 \\ 1-\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=0\end{cases} \\(9)
$$
对于任意 $m ≤ k$，如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 和 $n \in \bigcup_{h=1}^H \tilde{\mathcal{B}}_h(\mathbf{x})$ 都满足 $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$，则树模型 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是 top-$m$ 贝叶斯最优的。如果对于任意 $\mathbf{x} \in \mathcal{X}$ 和 $n ∈ \mathcal{N}$ 都满足$p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$，则 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是贝叶斯最优的。

定义1 展示了最优的树模型应该满足的一个情况，但它并没有解决所有问题，因为学习和评估树模型都需要对其次优性进行定量衡量。

注意，公式（8）意味着
$$
\mathbb{E}_{p(\mathbf{x})}\left[\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(k)}} \eta_j(\mathbf{x})\right]=\mathbb{E}_{p(\mathbf{x})}\left[\sum_{n \in \mathcal{B}_H(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right]
$$
其中 $\mathcal{I}_{\mathbf{x}}^{(k)}=\operatorname{argTopk}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$ 表示根据真实值 $η_j (x)$ 获取前 $k$ 个目标。这个方程的偏差可以用作 $\mathcal{M}$ 的次优性度量。正式地，我们将其定义为关于Precision@k的遗憾度，并将其表示为 $\operatorname{reg}_{p @ k}(\mathcal{M})$ 。 这是当 $m = k$ 时的一种特殊情况，对于更一般的定义 $\operatorname{reg}_{p @ k}(\mathcal{M})$，有 

$$
\mathbb{E}_{p(\mathbf{x})}\left[\frac{1}{m}\left(\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(m)}} \eta_j(\mathbf{x})-\sum_{n \in \mathcal{B}_H^{(m)}(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right)\right] \\(10)
$$
其中 $\mathcal{I}_{\mathbf{x}}^{(m)}=\operatorname{argTopm}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$。

尽管 $\operatorname{reg}_{p @ k}(\mathcal{M})$ 似乎是一个理想的次优性度量，但由于存在一系列嵌套的、不可微分的 argTopk 运算符，找到它的优化器是非常困难的。因此，找到一个替代损失函数 $\operatorname{reg}_{p @ k}(\mathcal{M})$，使得其最小化器仍然是一个最优的树模型，变得非常重要。为了区分这样的替代损失函数，我们引入了 beam search 下的校准概念，如下所示： 

**定义2**（beam search 下的校准性）。给定树模型 $\mathcal{M}(\mathcal{T}, g)$，一个损失函数 $L:\{0,1\}^M \times \mathbb{R}^{|\mathcal{N}|} \rightarrow \mathbb{R}$ 被称为在 beam search 下 top-$k$ 校准的，如果对于任意分布 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 都有 
$$
\underset{a}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}[L(\mathbf{y}, \mathbf{g}(\mathbf{x}))] \subset \underset{a}{\operatorname{argmin}} \operatorname{reg}_{p @ k}(\mathcal{M}) \\(11)
$$
对于任意 $1 ≤ k ≤ M$，如果公式（11）对于任意 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 都成立，则 $L$ 在 beam search下被称为校准的。

定义2展示了一般情况下，具有最小化非校准损失的树模型 $\mathcal{M}(\mathcal{T}, g)$ 在 beam search 下不是贝叶斯最优的。回顾命题1表明，对于任何 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 和任何 $\mathcal{T}$，$\operatorname{reg}_{p @ k}(\mathcal{M})$ 的最小化器总是存在，并且满足 $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$ 以及 $\operatorname{reg}_{p @ k}(\mathcal{M})=0$。因此，通过证明训练损失的最小化器通常不能保证 $\operatorname{reg}_{p @ k}(\mathcal{M})=0$，可以证明 TDMs 和 PLTs 的次优性。通过找到一个反例，以及在表1中展示的简单实验，我们满足了这个要求。因此，我们得到以下结论：

命题2. 公式(2)中定义的 $z_n$ 在一般情况下不是 beam search 下的校准的。

### 4.3. Learning Optimal Tree Models under Beam Search

根据第4.2节的讨论，我们需要一个新的替代损失函数，使得它的最小化器对应于在 beam search 下是贝叶斯最优的树模型。根据定义1，当使用Precision@m来衡量检索性能时，要求一个模型在 beam search 下是top-m的贝叶斯最优将足够。定义1提供了一个自然的替代损失函数来实现这个目的，其中 beam search 的大小 $k ≥ m$，即
$$
g \in \underset{g}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x})}\left[\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \operatorname{KL}\left(p^*\left(z_n \mid \mathbf{x}\right) \| p_g\left(z_n \mid \mathbf{x}\right)\right)\right] \\(12) 
$$
其中我们遵循 TDM 的风格，并假设 $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$。 

与公式（2）不同，公式（12）在训练中使用 $\tilde{\mathcal{B}}_h(\mathbf{x})$ 中的节点，而不是 $\mathcal{S}_h^{+}(\mathbf{y})$，并且相对于公式（1），引入了不同的伪目标定义。设 $z_n^* \sim p^*\left(z_n \mid \mathbf{x}\right)$ 表示相应的伪目标，我们有：
$$
z_n^*=y_{\pi\left(n^{\prime}\right)}, n^{\prime} \in \underset{n^{\prime} \in \mathcal{L}(n)}{\operatorname{argmax}} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}) \\(13)
$$
注意，对于 $n \in \mathcal{N}_H$，$z_n^*=y_{\pi(n)}$以及公式（1）中的 $z_n$ 也是一样的。为了区分 $z_n^*$ 和 $z_n$，我们将其称为最优伪目标，因为它对应于最优的树模型。

根据这个定义，公式（12）可以重写为 $\operatorname{argmin}_g \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))\right]$，其中：
$$
L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \ell_{\mathrm{BCE}}\left(z_n^*, g(\mathbf{x}, n)\right) \\(14)
$$
注意，在公式（14）中，我们为 $z^*_n$ 赋予了下标 $p$ 以突出其对 $η_j (x)$ 的依赖关系，这意味着公式（14）在 beam search 下是校准的，即其形成取决于 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{\top}$。

图1提供了 $z^*_n$ 和 $z_n$ 之间差异的具体示例。并非所有相关目标 $y_j = 1$ 的祖先节点都被视为相关节点，根据 $z^*_n$ ：节点1和6（图1（a）中的红色节点）被赋予 $z_n = 1$ 但 $z^*_n = 0$（图1（c）中的绿色节点）。原因是在以这些节点为根的子树上，不相关目标相对于相关目标具有更高的 $η_j (x)$，即 $\eta_7(\mathbf{x})=0.5>\eta_8(\mathbf{x})=0.4$ 和 $\eta_1(\mathbf{x})=0.8>\eta_3(\mathbf{x})=0.7$，这导致 $z^*_n$ 为0。 然而，直接最小化公式（14）是不可能的，因为在实践中无法得知 $η_j (\mathbf{x})$。因此，我们需要找到一个近似的 $z^*_n$，使其不依赖于 $η_j (\mathbf{x})$。假设 $g(\mathbf{x}, n)$ 的参数化使用可训练的参数 $\boldsymbol{\theta} \in \boldsymbol{\Theta}$，我们使用记号 $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$，$p_{g_{\boldsymbol{\theta}}}(\mathbf{x})$ 和 $\mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$ 来强调它们对 $θ$ 的依赖性。一个自然的选择是用 $p_{gθ} (z_{n'} = 1|\mathbf{x})$ 来替换公式（13）中的 $\eta_{\pi\left(n^{\prime}\right)}(\mathbf{x})$。然而，这个表达式仍然不实际，因为对于每个 $n \in \mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$，遍历 $\mathcal{L}(n)$ 的计算复杂度是不可接受的。由于树结构的存在，我们可以用 $$\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$$ 近似表示 $z_n^*$，对于 $n \in \mathcal{N} \backslash \mathcal{N}_H$，它以递归的方式构造如下： 
$$
\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=\hat{z}_{n^{\prime}}(\mathbf{x} ; \boldsymbol{\theta}), n^{\prime} \in \underset{n^{\prime} \in \mathcal{C}(n)}{\operatorname{argmax}} p_{g_{\boldsymbol{\theta}}}\left(z_{n^{\prime}}=1 \mid \mathbf{x}\right), \\(15)
$$
对于 $n \in \mathcal{N}_H$，直接设定为 $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=y_{\pi(n)}$。

这样做可以消除对未知 $η_j (x)$ 的依赖性。但是，当用 $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 替换 $z^*_n$ 时，最小化公式（14）仍然不是一个简单的任务，因为参数 $θ$ 会影响 $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta}) $、$\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 和 $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$ ：由于 $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta})$ 中的argTopk操作符以及 $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$ 中的argmax操作符的不可微性，无法直接计算关于 $θ$ 的梯度。为了得到一个可微的损失函数，我们建议用公式（14）中定义的 $L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))$ 替换为 
$$
L_{\boldsymbol{\theta}_t}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)} \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right) \\(16)
$$
其中 $θ_t$ 表示固定的参数，可以是梯度下降算法中的最后一次迭代的参数。根据上述讨论，我们提出了一种用于学习这样一个树模型的新算法，如算法1所示。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Alg1.png)

根据附录中的分析，算法1的训练复杂度为每个实例 $\mathrm{O}\left(H b k+H b\left|\mathcal{I}_{\mathbf{x}}\right|\right)$，仍然对 $M$ 是对数复杂度。此外，根据算法1训练的树模型，在测试时的复杂度为每个实例 $O(Hbk)$，与第3.2.2节中相同，因为算法1在测试中没有改变 beam search 的方式。

现在，剩下的问题是，由于在公式（16）中引入了几个近似，它是否仍然具有在 beam search 下实现贝叶斯最优性的良好性质？我们给出如下答案： 

命题3（实用算法）。假设 $\mathcal{G}=\left\{g_{\boldsymbol{\theta}}:\right.\boldsymbol{\theta} \in \boldsymbol{\Theta}\}$ 具有足够的容量，并且 $L_{\boldsymbol{\theta}_t}^*(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=$ 
$$
\sum_{h=1}^H \sum_{n \in \mathcal{N}_h} w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right) \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right) \\(17) 
$$
其中 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$。对于任意概率 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^+$，如果存在 $\boldsymbol{\theta}_t \in \boldsymbol{\Theta}$ 使得 
$$
\boldsymbol{\theta}_t \in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_{\boldsymbol{\theta}_t}^{\boldsymbol{*}}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})\right] \\(18)
$$
则相应的树模型 $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ 在 beam search 下是贝叶斯最优的。 

命题3表明，用 $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 替换 $z^*_n$ 并引入固定参数 $θ_t$ 不会影响公式（17）中的 $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ 的最优性。然而，公式（16）没有这样的保证，因为对 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 的求和对应于在权重 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)=\mathbb{I}\left(n \in \mathcal{B}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)\right)$ 下对 $\mathcal{N}_h$ 的求和，从而违反了 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$ 的限制。这个问题可以通过在公式（16）中引入随机性来解决，使得每个 $n \in \mathcal{N}_h$ 在期望情况下都有一个非零值。例如，将 $\mathcal{N}_h$ 的随机样本添加到公式（16）的求和中，或者利用随机 beam search (Kool等，2019)来生成 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$。然而，在实验中我们发现这些策略并不会对性能产生很大影响，因此我们仍然使用公式（16）。

> 译注：自底向上按照最优原则调整非叶节点。

## 5.Experiments

在本节中，我们通过实验证明了我们的分析，并评估了不同树模型在合成数据和真实数据上的性能。在整个实验过程中，我们使用OTM来表示根据算法1训练的树模型，因为它的目标是在 beam search 下学习最优的树模型。为了进行消融研究，我们考虑了OTM的两个变体：OTM (-BS)与OTM不同之处在于将 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 替换为 $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \cup \mathcal{S}_h^{-}(\mathbf{y})$；OTM (-OptEst)与OTM不同之处在于将公式（13）中的 $\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 替换为公式（1）中的 $z_n$。更多实验的详细信息可以在附录中找到。

### 5.1. Synthetic Data

**数据集**：对于每个实例 $(\mathbf{x}, \mathbf{y})$，$\mathbf{x} \in \mathbb{R}^d$ 是从零均值和单位协方差矩阵的 $d$ 维各向同性高斯分布 $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$ 中采样得到的，而 $\mathbf{y} \in \{0,1\}^M$ 是从 $p(\mathbf{y} \mid \mathbf{x})=\prod_{j=1}^M p\left(y_j \mid \mathbf{x}\right)=\prod_{j=1}^M 1 /\left(1+\exp \left(-\left(2 y_j-1\right) \mathbf{w}_j^{\top} \mathbf{x}-b\right)\right)$ 中采样得到的，其中权重向量 $\mathbf{w}_j \in \mathbb{R}^d$ 也是从 $\mathcal{N}\left(\mathbf{0}_d, \mathbf{I}_d\right)$ 中采样得到的。偏置 $b$ 是一个预定义的常数，用于控制 $\mathbf{y}$ 中非零条目的数量。相应的训练和测试数据集分别表示为 $D_{tr}$ 和 $D_{te}$。

**比较的模型和指标**：我们将OTM与PLT和TDM进行比较。所有的树模型 $\mathcal{M}(\mathcal{T}, g)$ 共享相同的树结构 $\mathcal{T}$ 和节点评分器 $g$ 的参数化。具体而言，$\mathcal{T}$ 被设置为在 $\mathcal{I}$ 上的随机二叉树，$g(\mathbf{x}, n)=\boldsymbol{\theta}_n^{\top} \mathbf{x}+b_n$ 被参数化为线性评分器，其中 $\boldsymbol{\theta}_n \in \mathbb{R}^d$ 和 $b_n \in \mathbb{R}$ 是可训练的参数。所有模型都在 $D_{tr}$ 上进行训练，并通过 $\widehat{\operatorname{reg}}_{p @ m}$ 来衡量它们的性能，该指标是将公式（10）中对 $p(\mathbf{x})$ 的期望替换为对 $\mathcal{D}_{t e}$ 中 $(\mathbf{x}, \mathbf{y})$ 的求和得到的。

**结果**：表2显示OTM相比于其他模型表现最好，这表明消除训练-测试差异可以提高树模型的检索性能。OTM (-BS)和OTM (-OptEst)的遗憾值比PLT和TDM小，这意味着仅使用 beam search 感知子采样（即$\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$）或估计的最优伪目标（即$\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$）单独对性能有所贡献。此外，OTM (-OptEst)的遗憾值比OTM (-BS)小，这表明 beam search 感知子采样对OTM的性能贡献大于估计的最优伪目标。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table2.png)

**Table 2**. A comparison of $\widehat{\operatorname{reg}}_{p @ m}(\mathcal{M})$ averaged by 5 runs with random initialization with hyperparameter settings $M = 1000$, $d = 10$, $b = −5$, $|D_{tr}| = 10000$, $|D_{te}| = 1000$ and $k = 50$.

### 5.2. Real Data

**数据集**：我们的实验是在两个大规模实际推荐任务数据集上进行的：Amazon图书数据集（McAuley等，2015；He＆McAuley，2016）和UserBehavior数据集（Zhu等，2018）。这两个数据集中的每条记录都以user-item 互动的方式组织，包含item ID、item ID和时间戳。原始的互动记录被表示为一组基于用户的数据。每个基于用户的数据被表示为按照 user-item 互动发生的时间步排序的 item 列表。我们丢弃了少于10个item 的基于用户的数据，并将其余数据集按照相同的方式分割为训练集$D_{tr}$、验证集$D_{val}$和测试集$D_{te}$。对于验证集和测试集，我们按照时间戳的升序顺序将每个基于用户的数据的前一半作为特征 $\mathbf{x}$，后一半作为相关目标 $y$。而训练实例是根据原始的基于用户的数据生成的，考虑到不同方法在训练集上的特点。如果方法限制$\left|\mathcal{I}_{\mathbf{x}}\right|=1$，我们使用一个滑动窗口为每个基于用户的数据生成多个实例，而对于没有$|\mathcal{I}_{\mathbf{x}}|$限制的方法，获得一个实例。

**比较的模型和指标**：我们将OTM与两类方法进行比较：（1）推荐任务中广泛使用的方法，例如Item-CF（Sarwar等，2001），基本的协同过滤方法，以及基于向量kNN的代表性作品YouTube productDNN（Covington等，2016）；（2）树模型，如HSM（Morin＆Bengio，2005），PLT和JTM（Zhu等，2019）。HSM是一种分层softmax模型，可以看作是具有$\left|\mathcal{I}_{\mathbf{x}}\right|=1$限制的PLT。JTM是TDM的一种变体，它同时训练树结构和节点评分器，并在这两个数据集上取得了最先进的性能。所有的树模型共享相同的二叉树结构，并采用相同的神经网络模型作为节点评分器。神经网络包含三个全连接层，隐藏大小分别为128、64和24，并使用参数化ReLU作为激活函数。不同模型的性能通过在测试集$D_{te}$上对Precision@m（公式（5））、Recall@m（公式（6））和F-Measure@m（公式（7））进行平均来衡量。 

**结果**：表3和表4分别显示了 Amazon图书 数据集和 UserBehavior数据集 的结果。我们的模型在所有方法中表现最好：与之前最先进的JTM相比，OTM在Amazon图书数据集和UserBehavior数据集上分别实现了29.8%和6.3%的相对召回率提升（m = 200）。OTM及其两个变体的结果与合成数据上的结果一致：beam search 感知子采样和估计的最优伪目标都对性能有所贡献，其中前者的贡献更大，OTM的性能主要依赖于前者。此外，HSM和PLT之间的比较也证明了在树模型中消除 $\left|\mathcal{I}_{\mathbf{x}}\right|=1$ 限制有助于提高性能。

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table3.png)

**Table 3**. Precision@m, Recall@m and F-Measure@m comparison on Amazon Books with beam size k = 400 and various m (%).

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table4.png)

**Table 4**. Precision@m, Recall@m and F-Measure@m comparison on UserBehavior with beam size k = 400 and various m (%).

为了理解为什么OTM在Amazon图书数据集上实现了比UserBehavior更显著的提升（29.8%对比6.3%），我们分析了这些数据集及其对应的树结构的统计信息。对于每个$n \in \mathcal{N}$，我们定义 $S_n=\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{tr}} z_n$ 来计算与 $n$ 相关的训练实例的数量（即$z_n=1$）。对于每个层级 $1 \leq h \leq H$，我们将 $\left\{S_n: n \in \mathcal{N}_h\right\}$ 按降序排序，并将它们归一化为 $S_n / \sum_{n^{\prime} \in \mathcal{N}_h} S_{n^{\prime}}$。这产生了一个按层级划分的分布，反映了由于数据集和树结构的固有属性而导致的相关节点上的数据不平衡情况。如图2所示，UserBehavior的按层级分布比Amazon图书更加尾重。这意味着后者有更高比例的实例集中在部分节点上，这使得 beam search 更容易检索到用于训练的相关节点，从而导致更显著的改进。 

为了验证我们关于树模型时间复杂度的分析，我们比较了它们的实证训练时间，因为它们在测试中共享相同的 beam search 过程。具体来说，我们计算了在UserBehavior数据集上使用批量大小为100进行训练的PLT、TDM和PLT的每个批次的墙钟时间。这个数字是在一块Tesla P100-PCIE-16GB GPU上的5000个训练迭代中平均得出的。结果分别为0.184秒的PLT，0.332秒的TDM和0.671秒的OTM。尽管OTM的时间比PLT和JTM更长，但它们具有相同数量级的时间复杂度。这并不奇怪，因为算法1中的第4步和第5步只增加了复杂性的常数因子。此外，这是为了获得更好的性能的合理权衡，在实际应用中可以通过分布式训练来减轻这个问题。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Figure2.png)

**Figure 2**. Results of the level-wise distribution versus node index on Amazon Books and UserBehavior with $h = 8 (|\mathcal{N}_h| = 256)$.

## 6. Conclusions and Future Work

树模型由于其对数级的计算复杂度，在大规模信息检索和推荐任务中得到了广泛应用。然而，对于训练-测试不一致性的问题，人们对训练中忽略了由于测试中的 beam search 导致的检索性能下降并没有给予足够重视。据我们所知，我们是第一个从理论上研究树模型中这个问题的人。我们还提出了一种新颖的训练算法，用于在 beam search 下学习最优的树模型，并在合成数据和真实数据上实现了比现有方法更好的实验结果。

对于未来的工作，我们希望探索其他根据方程（14）来训练$g(\mathbf{x}, n)$ 的技术，例如REINFORCE算法（Williams, 1992; Ranzato et al., 2016）和actor-critic算法（Sutton et al., 2000; Bahdanau et al., 2017）。我们还希望扩展我们的算法，同时学习树结构和节点评分器。此外，将我们的算法应用于极端多标签文本分类等应用也是一个有趣的方向。