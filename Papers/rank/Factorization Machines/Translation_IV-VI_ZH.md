## IV. FMS VS. SVMS

#### A. SVM model

支持向量机[6] 的模型方程可以表示为经转换的输入 $x$ 与模型参数 $\textbf{w}$ 的点积： $\hat{y}(\textbf{x}) = <\phi(\textbf{x})，\textbf{w}>$，其中 $\phi$ 是特征空间 $\mathbb{R}^n$ 到更复杂空间 $\mathcal{F}$ 的映射。映射 $\phi$ 与核函数有以下关系：
$$
K:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R},\quad K(\textbf{x},\textbf{z}) = <\phi(\textbf{x}),\phi(\textbf{z})>
$$
接下来，我们通过分析 SVMs3 的原始形式来讨论 FMs 和 SVMs 的关系。(实际操作上，支持向量机以对偶形式求解，并且不显式执行映射 $\phi$。然而，原始形式和对偶形式有相同的解(最优)，所以我们所有关于原始形式的讨论也适用于对偶形式。)

#### 1) Linear kernel:

最简单的核是线性核：$K_l(\textbf{x}，\textbf{z}) := 1+<\textbf{x}，\textbf{z}>$，对应于映射$\phi(X) := (1，x_1，\cdots，x_n)$。因此线性支持向量机的模型方程可以改写为：
$$
\hat{y}(\textbf{x})=w_0+\sum_{i=1}^{n}w_ix_i,
\quad w_0 \in \mathbb{R},
\quad \textbf{w} \in \mathbb{R}^n
\qquad(7)
$$
明显，线性核的 SVM (eq. (7)) 等价于 $d=1$ 的 FM (eq. (5)).

#### 2) Polynomial kernel:

多项式核允许支持向量机对变量之间更高阶的交互进行建模。定义为 $K(\textbf{x}，\textbf{z})：=(<\textbf{x}，\textbf{z}>+1)^d$。例如，对于 $d=2$ ，这对应于以下映射：
$$
\phi(\textbf{x}) := 
\\
(1,\sqrt{2}x_1,\cdots, \sqrt{2}x_n,
\\
x_1^2,\cdots,x_n^2,
\\
\sqrt{2}x_1x_2,\cdots,\sqrt{2}x_1x_n,
\\
\sqrt{2}x_2x_3,\cdots,\sqrt{2}x_{n-1}x_n
)
\qquad
(8)
$$
因此，polynomial 支持向量机的模型方程可以重写为：
$$
\hat{y}=w_0+\sqrt{2}\sum_{i=1}^{n}w_ix_i+\sum_{i=1}^{n}w_{i,i}^{(2)}x_i^2 
\\
+ \sqrt{2}\sum_{i=1}^{n}\sum_{j=i+1}^nw_{i,j}^{(2)}x_ix_j
\qquad(9)
$$
其中，模型参数为：
$$
w_0 \in \mathbb{R} \quad , w \in \mathbb{R}^n,
\quad \textbf{W}^{(2)} \in \mathbb{R}^{n \times n}
(symmetric \ matrix)
$$
比较多项式的 支持向量机 (eq . (9)) 和 FM (eq. (1))，可以看到这两个模型都对次数为 $d=2$ 的交互建模。SVMs 和 FMs 的主要区别在于参数化：支持向量机的所有交互参数 $w_{i，j}$ 都是完全独立的，例如 $w_{i，j}$ 和 $w_{i，l}$。与此相反， FM 的交互参数被分解，因此 $<\textbf{v}_i，\textbf{v}_j>$ 和 $<\textbf{v}_i，\textbf{v}_l>$ 相互依赖，因为它们重叠并共享参数(这里为 $\textbf{v}_i$ )。

#### B. Parameter Estimation Under Sparsity

下面，我们将展示为什么线性和多项式支持向量机对于非常稀疏的问题会失败。我们在使用 user 和 item 的指示变量的协同过滤示例中展示了这一点(参见 图1 示例中的前两组(蓝色和红色))。这里，特征向量是稀疏的，并且只有两个元素是非零的(user $u$ 和 item $i$)。

**1. Linear SVM：**

对于这类数据 $\textbf{x}$，线性支持向量机模型(Eq.(7))相当于：
$$
\hat{y}(\mathbf{x})=w_{0}+w_{u}+w_{i} \qquad(10)
$$
因为 $x_j = 1$ 当且仅当 $j = u$ 或 $j = i$ 。这个模型对应于一个最基本的协同过滤模型，其中只能捕获用户和物品的偏差。由于该模型非常简单，即使在稀疏条件下也能很好地估计参数。然而，经验预测的质量通常较低(见图2)。

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Factorization Machines/Fig2.png)

**图2 . 在支持向量机失败的非常稀疏的问题中，FMS 成功地估计了双变量交互(有关详细信息，请参阅第III-A3和IV-B部分)。**

**2. Polynomial SVM: **

通过多项式核，支持向量机可以捕获更高阶的交互(这里是 user 和 item 之间的交互)。在 $m(x) = 2$ 的稀疏情况下，支持向量机的模型方程等价为:
$$
\hat{y}(\textbf{x})=w_0+\sqrt{2}(w_u+w_i) + w_{u,u}^{(2)}+w_{i,i}^{(2)} + \sqrt{2}w_{u,i}^{(2)}
$$
首先，$w_u$ 和 $w^{(2)}_{u，u}$ 表示相同-即可以删除其中之一(例如，$w^{(2)}_{u，u}$)。现在，模型方程与线性情况相同，但增加了 user-item 交互 $w^{(2)}_{u，i}$。

在典型的协同过滤(CF)问题中，对于每个交互参数 $w^{(2)}_{u,i}$ ，训练数据中至多有一个观察值 $(u，i)$，而对于测试数据中的 $(u^{’}，i^{’})$，训练数据中通常根本没有观察值。

例如，在 图1 中，只有一个交互(Alice, Titanic) 有观察值，而交互 (Alice, StarTrek) 没有观察值。这意味着对于所有测试用例 $(u, i)$ ，交互参数 $w^{(2)}_{u,i}$ 的 maximum margin 解决方案是 $0$ (例如 $w^{(2)}_{A,ST} = 0$ )。因此，多项式支持向量机不能利用任何双交互来预测测试集示例；因此多项式支持向量机仅依赖于 user 和 item 的偏差，并不能提供比线性支持向量机更好的估计。

对于支持向量机来说，估计高阶交互不仅是 CF 中的一个问题，而且在数据非常稀疏的所有场景中都是一个问题。因为对于两两交互 $(i，j)$ 的参数 $w^{(2)}_{i，j}$ 的可靠估计，必须有“足够”的样本 $\textbf{x} \in D$，其中 $x_{i} \neq 0 \wedge x_{j} \neq 0$。只要 $x_i=0$ 或 $x_j=0$ ，样本 $\textbf{x}$ 就不能用于估计参数 $w^{(2)}_{i，j}$。总而言之，如果数据太稀疏，即 $(i，j)$ 的案例太少，甚至没有案例，则支持向量机很可能失败。

## C. Summary

1. 支持向量机的密集参数化要求直接观察值变量的交互，这在稀疏环境下通常不能给出。FM 即使在稀疏条件下，也可以很好地估计参数（请参阅第III-A3节）。
2. FMs 可以直接在原始阶段学习。非线性支持向量机通常是在对偶中学习的。
3. FMs 的模型方程与训练数据无关。支持向量机的预测依赖于部分训练数据(支持向量)。

## V. FMs VS. OTHER FACTORIZATION MODELs

有各种各样的因子分解模型，从分类变量 m-ary 关系的标准模型(如 MF, PARAFAC )到针对特定数据和任务的专门模型(如 SVD++， PITF, FPMC )。接下来，我们展示了 FM 只需使用正确的输入数据(例如，特征向量 $\textbf{x}$ )就可以模拟其中的许多模型。

#### A. Matrix and Tensor Factorization

矩阵分解(MF)是目前研究最多的分解模型之一(如[7]、[8]、[2])。它分解了两个类别变量(例如 $U$ 和 $I$ )之间的关系。处理类别变量的标准方法是为 $U$ 和 $I$ 的每个级别定义二进制指示符变量(例如，见图2)。第一组(蓝色)和第二组(红色))4：
$$
n:=|U \cup I|, \quad x_{j}:=\delta(j=i \vee j=u)
\qquad(11)
$$
(为了简化表示法，我们在 $\textbf{x}$ (例如 $x_j$) 和参数中都使用数字( 例如$j∈\{1,\cdots,n\}$ )，类别 ( 例如 $j∈(U∪I)$ ) 中的元素。这意味着我们隐含地假设了一个从 数值 到 类别 的双射映射。)

FM 使用特征向量 $\textbf{x}$ 与矩阵分解模型[2]相同，因为 $x_j$ 对于 $u$ 和 $i$ 只是非零值，因此所有其他偏差和交互作用都丢弃掉：
$$
\hat{y}(\textbf{x}) = w_0 + w_u + w_i + <\textbf{v}_u,\textbf{v}_i>
\qquad(12)
$$
用相同的论点，可以看到，对于具有两个以上类别变量的问题，FM 包括嵌套的并行因子分析模型（PARAFAC）[1]。

#### B. SVD++

对于评分预测(即回归)，Koren将矩阵分解模型改进为 SVD++ 模型[2]。
FM 可以通过使用输入数据 $\textbf{x}$ 来模拟此模型(如图1的前三组中所示)：
$$
n:=|U \cup I \cup L|, 
\quad x_{j}:=
\left\{\begin{array}{ll}
1, & \text{ if } j=i \vee j=u \\
\frac{1}{\sqrt{\left|N_{u}\right|}}, & \text { if } j \in N_{u} \\
0, & \text { else }
\end{array}\right.
$$
(为区别 $N_u$ 和 $I$ 中的元素，将元素用任意双射函数 $ω: I→L$ 变换为空间 $L$ ， $L∩I =∅$。)

其中 $N_u$ 是用户曾经评分过的所有电影的集合5。使用此数据，FM ($d=2$)将表现如下：
$$
\begin{array}{c}
\hat{y}(\mathbf{x})=
\overbrace{w_{0}+w_{u}+w_{i}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{i}\right\rangle+
\frac{1}{\sqrt{\left|N_{u}\right|}} \sum_{l \in N_{u}}\left\langle\mathbf{v}_{i},\mathbf{v}_{l}\right\rangle}^{\text {SVD++ }} 
\\
+\frac{1}{\sqrt{\left|N_{u}\right|}} \sum_{l \in N_{u}}
\left(w_{l}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{l}\right\rangle+\frac{1}{\sqrt{\left|N_{u}\right|}} \sum_{l^{\prime} \in N_{u}, l^{\prime}>l}\left\langle\mathbf{v}_{l}, \mathbf{v}_{l}^{\prime}\right\rangle\right)
\end{array}
$$
其中第一部分与 SVD++ 模型完全相同但是 FM 还包含用户与电影 $N_u$ 之间的一些额外交互，以及电影 $N_u$ 的基本影响以及 $N_u$ 中电影对之间的交互。

#### C. PITF for Tag Recommendation

标签预测问题被定义为对给定 user 和 item 组合的标签进行排序。这意味着涉及三个类别领域：user $U$、item $I$ 和 tag $T$。在 ECML/PKDD 标签推荐发现挑战赛中，基于因子分解对交互(PITF)的模型取得了最好的成绩[3]。

我们将演示 FM 如何模仿这个模型。具有激活的 user $u$、item $i$ 和 tag $t$ 的二进制指示符变量的因子分解机产生以下模型：
$$
\begin{aligned}
n &:=|U \cup I \cup T|, \quad x_{j}:=\delta(j=i \vee j=u \vee j=t) \\
\Rightarrow \ & \hat{y}(\mathbf{x})=w_{0}+w_{u}+w_{i}+w_{t}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{i}\right\rangle+\left\langle\mathbf{v}_{u}, \mathbf{v}_{t}\right\rangle+\left\langle\mathbf{v}_{i}, \mathbf{v}_{t}\right\rangle
\end{aligned}
\qquad(13)
$$
由于该模型用于同一 user/item 组合 $(u, i)$ [3]中两个标签 $t_A$， $t_B$ 之间的排序，因此，无论是优化还是预测，都是基于样本 $(u, i, t_A)$ 和 $(u, i, t_B)$ 的得分差异。因此，对 pair-wise 排序进行优化(如[5]、[3])，FM 模型等价为：
$$
\hat{y}(\mathbf{x}):=w_{t}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{t}\right\rangle+\left\langle\mathbf{v}_{i}, \mathbf{v}_{t}\right\rangle
\qquad(14)
$$
现在，原始的 PITF 模型[3]和带有二元指标的 FM 模型(等式，(14)几乎完全相同。不同的是：

1. FM 模型对 $t$ 有偏置项 $w_t$。
2. $(u，t)$ 和 $(i，t)$ 交互之间的标签 ($\textbf{v}_t$) 的因子分解参数对于 FM 模型是共享的，但对于原始 PITF 模型是独立的。

除了理论分析外，图3 经验上表明两种模型对该任务的预测质量也具有可比性的。

#### D. Factorized Personalized Markov Chains (FPMC)

FPMC 模型[4]尝试根据 user $u$ 的最后一次购买(时间 $t−1$ )对线上商店中的产品进行排序。

同样地，通过特征生成，分解机 ($d = 2$)的行为也类似：
$$
n:=|U \cup I \cup L|, \quad x_{j}:=\left\{\begin{array}{ll}
1, & \text { if } j=i \vee j=u \\
\frac{1}{\left|B_{t-1}^{u}\right|}, & \text { if } j \in B_{t-1}^{u} \\
0, & \text { else }
\end{array}\right.
\qquad(15)
$$
其中 $B^u_t⊆L$ 是用户 $u$ 在时间 $t$ 购买的所有 item 的集合(有关详细信息，请参阅[4])。然后：
$$
\begin{array}{c}
\hat{y}(\mathbf{x})=w_{0}+w_{u}+w_{i}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{i}\right\rangle+\frac{1}{\left|B_{t-1}^{u}\right|} \sum_{l \in B_{t-1}^{u}}\left\langle\mathbf{v}_{i}, \mathbf{v}_{l}\right\rangle \\
+\frac{1}{\left|B_{t-1}^{u}\right|} \sum_{l \in B_{t-1}^{u}}\left(w_{l}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{l}\right\rangle+\frac{1}{\left|B_{t-1}^{u}\right|} \sum_{l^{\prime} \in B_{t-1}^{u}, l^{\prime}>l}\left\langle\mathbf{v}_{l}, \mathbf{v}_{l}^{\prime}\right\rangle\right)
\end{array}
$$
与标签推荐一样，该模型用于排序并对其进行优化(这里对 item $i$ 进行排序)，因此在预测和优化准则中仅使用 $(u，i_A，t)$ 和 $(u，i_B，t)$ 之间的分数差[4]。因此，所有不依赖于 $i$ 的附加项消失，FM 模型方程等价于：
$$
\hat{y}(\mathbf{x})=w_{i}+\left\langle\mathbf{v}_{u}, \mathbf{v}_{i}\right\rangle+\frac{1}{\left|B_{t-1}^{u}\right|} \sum_{l \in B_{t-1}^{u}}\left\langle\mathbf{v}_{i}, \mathbf{v}_{l}\right\rangle
\qquad(16)
$$
现在可以看到，原始的 FPMC 模型[4]和 FM 模型几乎相同，仅在 $(u，i)$ -和 $(i，l)$ - 交互中的附加偏置项 $w_i$ 和 FM 模型的因子分解参数的共享方面有所不同，而不同之处在于：对于 $(u，i)$ -和 $(i，l)$ -交互的 item，FM 模型的因子分解参数是共享的。

#### E. Summary

1. 像 PARAFAC 或 MF 这样的标准因子分解模型不是像 FM 那样的通用预测模型。相反，它们要求将特征向量分成 $m$ 部分，并且在每个部分中恰好有一个元素是 $1$ ，其余的是 $0$ (one-hot)。
2. 针对单一任务设计的专门化因子分解模型有的很多。我们已经证明，仅通过特征提取，因子分解机就可以模仿许多最成功的因子分解模型(包括 MF、PARAFAC、SVD++、PITF、FPMC)，这使得 FM 易于在实践中应用。

## VI. CONCLUSION AND FUTURE WORK

在这篇文章中，我们引入了因子分解机。FMs 结合了支持向量机的通用性和因子分解模型的优点。与支持向量机相比：

1. FMs 能够在极大的稀疏性下估计参数。
2. 模型方程是线性的，并且仅依赖于模型参数。
3. 可以直接在原始形式优化

FMs 的表达能力与多项式支持向量机相当。与 PARAFAC 等张量分解模型不同，FM 是一种可以处理任何实值向量的通用预测器。

此外，通过简单地在输入特征向量中使用正确的指示符，FMs 与许多仅适用于特定任务的专门的最先进的模型相同或非常相似，其中包括 biased MF、SVD++、PITF 和 FPMC。

## REFERENCES

 [1] R. A. Harshman, “Foundations of the parafac procedure: models and conditions for an ’exploratory’ multimodal factor analysis.” UCLA Working Papers in Phonetics, pp. 1–84, 1970. 

[2] Y. Koren, “Factorization meets the neighborhood: a multifaceted collaborative filtering model,” in KDD ’08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2008, pp. 426–434. 

[3] S. Rendle and L. Schmidt-Thieme, “Pairwise interaction tensor factorization for personalized tag recommendation,” in WSDM ’10: Proceedings of the third ACM international conference on Web search and data mining. New York, NY, USA: ACM, 2010, pp. 81–90. 

[4] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, “Factorizing personalized markov chains for next-basket recommendation,” in WWW ’10: Proceedings of the 19th international conference on World wide web. New York, NY, USA: ACM, 2010, pp. 811–820. 

[5] T. Joachims, “Optimizing search engines using clickthrough data,” in KDD ’02: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2002, pp. 133–142. 

[6] V. N. Vapnik, The nature of statistical learning theory. New York, NY, USA: Springer-Verlag New York, Inc., 1995. 

[7] N. Srebro, J. D. M. Rennie, and T. S. Jaakola, “Maximum-margin matrix factorization,” in Advances in Neural Information Processing Systems 17. MIT Press, 2005, pp. 1329–1336. 

[8] R. Salakhutdinov and A. Mnih, “Bayesian probabilistic matrix factorization using Markov chain Monte Carlo,” in Proceedings of the International Conference on Machine Learning, vol. 25, 2008.

----------------

## QA

1. SVM 和 FM 的主要区别

   > 4


