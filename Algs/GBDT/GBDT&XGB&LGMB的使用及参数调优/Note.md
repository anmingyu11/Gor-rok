> https://zhuanlan.zhihu.com/p/33700459

# GBDT、XGBoost、LightGBM 的使用及参数调优

## **GBDT**

## 概述

GBDT 是梯度提升树（Gradient Boosting Decison Tree）的简称，GBDT 也是集成学习 Boosting 家族的成员，但是却和传统的 Adaboost 有很大的不同。回顾下 Adaboost，我们是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。GBDT 也是迭代，使用了前向分布算法，同时迭代思路和 Adaboost 也有所不同。

GBDT 通过多轮迭代，每轮迭代产生一个弱分类器，每个分类器在上一轮分类器的残差基础上进行训练。对弱分类器的要求一般是足够简单，并且是低方差和高偏差的。因为训练的过程是通过降低偏差来不断提高最终分类器的精度。

弱分类器一般会选择为 CART（也就是分类回归树）。由于上述高偏差和简单的要求，每个分类回归树的深度不会很深。最终的总分类器是将每轮训练得到的弱分类器加权求和得到的（也就是加法模型）。

让损失函数沿着梯度方向的下降就是 GBDT 的核心了。利用损失函数的负梯度在当前模型的值作为回归问题提升树算法中的残差的近似值去拟合一个回归树。GBDT 每轮迭代的时候，都去拟合损失函数在当前模型下的负梯度。

算法如下（截图来自《The Elements of Statistical Learning》）：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/GBDT&XGB&LGMB的使用及参数调优/Fig1.png)

算法步骤解释：

1. 初始化，估计使损失函数极小化的常数值，它是只有一个根节点的树，即 ganma 是一个常数值。
2. 迭代
   a. 计算损失函数的负梯度在当前模型的值，将它作为残差的估计
   b. 估计回归树叶节点区域，以拟合残差的近似值
   c. 利用线性搜索估计叶节点区域的值，使损失函数极小化
   d. 更新回归树
3. 得到输出的最终模型 $f(x)$

下面我们具体来说 CART (是一种二叉树) 如何生成。CART 生成的过程其实就是一个选择特征的过程。假设我们目前总共有 $M$ 个特征。第一步我们需要从中选择出一个特征 $j$，做为二叉树的第一个节点。然后对特征 $j$ 的值选择一个切分点 $m$。一个 样本的特征 $j$ 的值 如果小于 $m$，则分为一类，如果大于 $m$，则分为另外一类。如此便构建了 CART 树的一个节点。其他节点的生成过程和这个是一样的。

## 参数说明（sklearn）

- n_estimators：控制弱学习器的数量。
- max_depth：设置树深度，深度越大可能过拟合。
- max_leaf_nodes：最大叶子节点数。
- learning_rate：更新过程中用到的收缩步长，(0, 1]。
- max_features：划分时考虑的最大特征数，如果特征数非常多，我们可以灵活使用其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
- min_samples_split：内部节点再划分所需最小样本数，这个值限制了子树继续划分的条件，如果某节点的样本数少于 min_samples_split，则不会继续再尝试选择最优特征来进行划分。
- min_samples_leaf：叶子节点最少样本数，这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。
- min_weight_fraction_leaf：叶子节点最小的样本权重和，这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。
- min_impurity_split：节点划分最小不纯度，使用 min_impurity_decrease 替代。
- min_impurity_decrease：如果节点的纯度下降大于了这个阈值，则进行分裂。
- subsample：采样比例，取值为(0, 1]，注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于 1，则只有一部分样本会去做 GBDT 的决策树拟合。选择小于 1 的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低，一般在 [0.5, 0.8] 之间。

回归树基学习器的大小定义了可以被梯度提升模型捕捉到的变量（即特征）相互作用（即多个特征共同对预测产生影响）的程度。 通常一棵深度为 $h$ 的树能捕获到秩为 $h$ 的相互作用。这里有两种控制单棵回归树大小的方法。

如果你指定 `max_depth = h` ，那么将会产生一个深度为 $h$ 的完全二叉树。这棵树将会有（至多） $2h$ 个叶子节点和 $2h - 1$ 个切分节点。

另外，你能通过参数 `max_leaf_nodes` 指定叶子节点的数量来控制树的大小。在这种情况下，树将会使用最优优先搜索来生成，这种搜索方式是通过每次选取对不纯度提升最大的节点来展开。一棵 `max_leaf_nodes = k` 的树拥有 $k - 1$ 个切分节点，因此可以模拟秩最高达到 `max_leaf_nodes - 1` 的相互作用（即 `max_leaf_nodes - 1` 个特征共同决定预测值）。

## 常见问题

**随机森林（random forest）和 GBDT 都是属于集成学习（ensemble learning）的范畴，有什么不同？**

集成学习下有两个重要的策略 Bagging 和 Boosting，Bagging 算法是这样，每个分类器都随机从原样本中做有放回的采样，然后分别在这些采样后的样本上训练分类器，然后再把这些分类器组合起来，简单的多数投票一般就可以，其代表算法是随机森林。Boosting 的算法是这样，它通过迭代地训练一系列的分类器，每个分类器采用的样本分布都和上一轮的学习结果有关。其代表算法是 AdaBoost，GBDT。

**为什么随机森林的树深度往往大于 GBDT 的树深度？**

其实就机器学习算法来说，其泛化误差可以分解为两部分，偏差（bias）和方差（variance）。偏差指的是算法的期望预测与真实预测之间的偏差程度，反应了模型本身的拟合能力；方差度量了同等大小的训练集的变动导致学习性能的变化，刻画了数据扰动所导致的影响。

如下图所示，当模型越复杂时，拟合的程度就越高，模型的训练偏差就越小。但此时如果换一组数据可能模型的变化就会很大，即模型的方差很大。所以模型过于复杂的时候会导致过拟合。

当模型越简单时，即使我们再换一组数据，最后得出的学习器和之前的学习器的差别就不那么大，模型的方差很小。还是因为模型简单，所以偏差会很大。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/GBDT&XGB&LGMB的使用及参数调优/Fig2.png)

也就是说，当我们训练一个模型时，偏差和方差都得照顾到，漏掉一个都不行。
对于 Bagging 算法来说，由于我们会并行地训练很多不同的分类器的目的就是降低这个方差（variance），因为采用了相互独立的基分类器多了以后，$h$ 的值自然就会靠近。所以对于每个基分类器来说，目标就是如何降低这个偏差（bias），所以我们会采用深度很深甚至不剪枝的决策树。

对于 Boosting 来说，每一步我们都会在上一轮的基础上更加拟合原数据，所以可以保证偏差（bias），所以对于每个基分类器来说，问题就在于如何选择 variance 更小的分类器，即更简单的分类器，所以我们选择了深度很浅的决策树。

**GBDT 如何用于分类？**

GBDT 无论用于分类还是回归一直都是使用的 CART 回归树。不会因为我们所选择的任务是分类任务就选用分类树，这里面的核心是因为 GBDT 每轮的训练是在上一轮的训练的残差基础之上进行训练的。这里的残差就是当前模型的负梯度值 。这个要求每轮迭代的时候，弱分类器的输出的结果相减是有意义的，残差相减是有意义的。

在分类训练的时候，是针对样本 $X$ 每个可能的类都训练一个分类回归树。针对样本有三类的情况，我们实质上是在每轮的训练的时候是同时训练三颗树。第一棵树针对样本 $x$ 的第一类，输入为 $(x, 0)$。第二棵树输入针对样本 $x$ 的第二类，假设 x 属于第二类，输入为 $(x, 1)$ 。第三棵树针对样本 $x$ 的第三类，输入为 $(x, 0)$ 。在这里每棵树的训练过程其实就是就是我们之前已经提到过的 CART 的生成过程。在此处我们参照之前的生成树的程序即可以就解出三棵树，以及三棵树对 $x$ 类别的预测值 $f_1(x), f_2(x), f_3(x)$。那么在此类训练中，我们仿照多分类的逻辑回归，使用 softmax 来产生概率。并且我们我们可以针对类别 $1$ 求出残差 $f_{1,1}(x) = 0 − f_1(x)$；类别 $2$ 求出残差 $f_{2,2}(x) = 1 − f_2(x)$；类别 $3$ 求出残差 $f_{3,3}(x) = 0 − f_3(x)$。然后开始第二轮训练，针对第一类输入为 $(x, f_{1,1}(x))$，针对第二类输入为$(x, f_{2,2}(x))$，针对第三类输入为 $(x, f_{3,3}(x))$。继续训练出三棵树，一直迭代 $M$ 轮，每轮构建 $3$ 棵树。当训练完毕以后，新来一个样本 $x_1$，我们需要预测该样本的类别的时候，便可使用 softmax 计算每个类别的概率。

## **XGBoost**

## 概述

XGBoost 是 “Extreme Gradient Boosting” 的缩写，XGBoost 算法的步骤和 GBDT 基本相同，都是首先初始化为一个常数，GBDT 是根据一阶导数，XGBoost 是根据一阶导数 $g_i$ 和二阶导数 $h_i$，迭代生成基学习器，相加更新学习器。

泰勒公式是一个用函数在某点的信息描述其附近取值的公式。基本形式是：
$$
f(x)=\sum_{n=0}^{\infty} \frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right)^{n}
$$
一阶泰勒展开：
$$
f(x) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)
$$
二阶泰勒展开：
$$
f(x) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+f^{\prime \prime}\left(x_{0}\right) \frac{\left(x-x_{0}\right)^{2}}{2}
$$
XGBoost 的损失函数不仅使用到了一阶导数，还使用二阶导数。
$$
\mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathrm{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
$$
对上述损失函数做二阶泰勒展开，其中 $g$ 为 一阶导数，$h$ 为二阶导数，最后一项为正则项，
$$
\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$

$$
\text { where } g_{i}=\partial_{\hat{y}(t-1)} l\left(y_{i}, \hat{y}^{(t-1)}\right) \text { and } h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)
$$

XGBoost 对分类前后的增益计算采用了如下方式（ID3 采用信息增益，C4.5 采用信息增益比，CART 采用 Gini 系数），

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/GBDT&XGB&LGMB的使用及参数调优/Fig3.png)

这个公式形式上跟 ID3 算法、CART 算法是一致的，都是用分裂后的某种值减去分裂前的某种值，从而得到增益。为了限制树的生长，我们可以加入阈值，当增益大于阈值时才让节点分裂，上式中的 gamma 即阈值，它是正则项里叶子节点数 T 的系数，所以 XGBoost 在优化目标函数的同时相当于做了预剪枝。另外，上式中还有一个系数 lambda，是正则项里 leaf score 的 L2 模平方的系数，对 leaf score 做了平滑，也起到了防止过拟合的作用，这个是传统 GBDT 里不具备的特性。

## 优势

1. 在寻找最佳分割点时，考虑传统的枚举每个特征的所有可能分割点的贪心法效率太低，XGBoost 实现了一种近似的算法。大致的思想是根据百分位法列举几个可能成为分割点的候选者，然后从候选者中根据上面求分割点的公式计算找出最佳的分割点。同时当分裂时遇到一个负损失时，GBM 会停止分裂。XGBoost 会一直分裂到指定的最大深度(max_depth)，然后回过头来剪枝。如果某个节点之后不再有正值，它会去除这个分裂。这种做法的优点，当一个负损失（如-2）后面有个正损失（如+10）的时候，就显现出来了。GBM 会在-2 处停下来，因为它遇到了一个负值。但是 XGBoost 会继续分裂，然后发现这两个分裂综合起来会得到+8，因此会保留这两个分裂。
2. 标准 GBM 的实现没有像 XGBoost 这样的正则化步骤。正则化对减少过拟合也是有帮助的。
3. XGBoost 考虑了训练数据为稀疏值的情况，可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率。
4. 列抽样，XGboost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。
5. 特征列排序后以块的形式存储在内存中，在迭代中可以重复使用；虽然 boosting 算法迭代必须串行，但是在处理每个特征列时可以做到并行。
6. 按照特征列方式存储能优化寻找最佳的分割点，但是当以行计算梯度数据时会导致内存的不连续访问，严重时会导致 cache miss，降低算法效率。paper 中提到，可先将数据收集到线程内部的 buffer，然后再计算，提高算法的效率。
7. XGBoost 还考虑了当数据量比较大，内存不够时怎么有效的使用磁盘，主要是结合多线程、数据压缩、分片的方法，尽可能的提高算法的效率。

## 参数说明

括号内为 sklearn 参数。

**通用参数**

- booster：基学习器类型，gbtree，gblinear 或 dart（增加了 Dropout） ，gbtree 和 dart 使用基于树的模型，而 gblinear 使用线性模型
- silent：使用 0 会打印更多信息
- nthread：运行时线程数

**Booster 参数**

树模型

- eta（learning_rate）：更新过程中用到的收缩步长，(0, 1]。
- gamma：在节点分裂时，只有在分裂后损失函数的值下降了，才会分裂这个节点。Gamma 指定了节点分裂所需的最小损失函数下降值。这个参数值越大，算法越保守。
- max_depth：树的最大深度，这个值也是用来避免过拟合的
- min_child_weight：决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。
- max_delta_step：这参数限制每颗树权重改变的最大步长。如果是 0 意味着没有约束。如果是正值那么这个算法会更保守，通常不需要设置。
- subsample：这个参数控制对于每棵树，随机采样的比例。减小这个参数的值算法会更加保守，避免过拟合。但是这个值设置的过小，它可能会导致欠拟合。
- colsample_bytree：用来控制每颗树随机采样的列数的占比。
- colsample_bylevel：用来控制的每一级的每一次分裂，对列数的采样的占比。
- lambda（reg_lambda）：L2 正则化项的权重系数，越大模型越保守。
- alpha（reg_alpha）：L1 正则化项的权重系数，越大模型越保守。
- tree_method：树生成算法，auto, exact, approx, hist, gpu_exact, gpu_hist
- scale_pos_weight：各类样本十分不平衡时，把这个参数设置为一个正数，可以使算法更快收敛。典型值是 sum(negative cases) / sum(positive cases)

Dart 额外参数

- sample_type：采样算法
- normalize_type：标准化算法
- rate_drop：前置树的丢弃率，有多少比率的树不进入下一个迭代，[0, 1]
- one_drop：设置为 1 的话每次至少有一棵树被丢弃。
- skip_drop：跳过丢弃阶段的概率，[0, 1]，非零的 skip_drop 比 rate_drop 和 one_drop 有更高的优先级。

线性模型

- lambda（reg_lambda）：L2 正则化项的权重系数，越大模型越保守。
- alpha（reg_alpha）：L1 正则化项的权重系数，越大模型越保守。
- lambda_bias（reg_lambda_bias）：L2 正则化项的偏置

**学习任务参数**

- objective：这个参数定义需要被最小化的损失函数。
- base_score：初始化预测分数，全局偏置。
- eval_metric：对于有效数据的度量方法，取值范围取决于 objective。
- seed：随机数种子，相同的种子可以复现随机结果，用于调参。

## **LightGBM**

## 概述

LightGBM 是微软开发的一款快速、分布式、高性能的基于决策树的梯度 Boosting 框架。主要有以下优势：

- 更快的训练效率
- 低内存使用
- 更好的准确率（我对比 XGBoost 没太大差别）
- 支持并行学习
- 可处理大规模数据

## 改进

**基于 Histogram 的决策树算法**

把连续的浮点特征值离散化成 $k$ 个整数，同时构造一个宽度为 $k$ 的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。
当然， histogram 算法也有缺点，它不能找到很精确的分割点，训练误差没有 pre-sorted 好。但从实验结果来看， histogram 算法在测试集的误差和 pre-sorted 算法差异并不是很大，甚至有时候效果更好。实际上可能决策树对于分割点的精确程度并不太敏感，而且较“粗”的分割点也自带正则化的效果。

**直方图做差加速**

一个叶子的直方图可以由它的父亲节点的直方图与它兄弟节点的直方图做差得到，提升一倍速度。

**带深度限制的 Leaf-wise 的叶子生长策略**

Level-wise 过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上 Level-wise 是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

Leaf-wise 则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同 Level-wise 相比，在分裂次数相同的情况下，Leaf-wise 可以降低更多的误差，得到更好的精度。Leaf-wise 的缺点是可能会长出比较深的决策树，产生过拟合。因此 LightGBM 在 Leaf-wise 之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

**直接支持类别特征(Categorical Feature)**

LightGBM 优化了对类别特征的支持，可以直接输入类别特征，不需要额外的 0/1 展开，并在决策树算法上增加了类别特征的决策规则。

**基于直方图的稀疏特征优化**

对于稀疏特征，只需要 O(2 * #non_zero_data) 来构建直方图。

**多线程优化**

在特征并行算法中，通过在本地保存全部数据避免对数据切分结果的通信。在数据并行中使用分散规约(Reduce scatter)把直方图合并的任务分摊到不同的机器，降低通信和计算，并利用直方图做差，进一步减少了一半的通信量。基于投票的数据并行(Parallel Voting)则进一步优化数据并行中的通信代价，使通信代价变成常数级别。特征并行的主要思想是在不同机器在不同的特征集合上分别寻找最优的分割点，然后在机器间同步最优的分割点。数据并行则是让不同的机器先在本地构造直方图，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点。

## 参数说明

XGBoost 和 LightGBM 参数对比

XGBoost / LightGBM

booster(default=gbtree) / boosting(default=gbdt)

eta(default=0.3) / learning_rate(default=0.1)

max_depth(default=6) / num_leaves(default=31)

min_child_weight(default=1) / min_sum_hessian_in_leaf(1e-3)

gamma(default=0) / min_gain_to_split(default=20)

subsample(default=1) / bagging_fraction(default=1.0)

colsample_bytree(default=1) / feature_fraction( default=1.0)

alpha(default=0) / lambda_l1(default=0)

lambda(default=1) / lambda_l2(default=0)

objective( default=reg:linear) / application(default=regression)

eval_metric(default according to objective) / metric

nthread / num_threads

括号内为 sklearn 参数。

- application（objective）：学习目标和损失函数。
- boosting（boosting_type）：‘gbdt’, traditional Gradient Boosting Decision Tree；‘dart’, Dropouts meet Multiple Additive Regression Trees；‘goss’, Gradient-based One-Side Sampling；‘rf’, Random Forest
- num_leaves：因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。大致换算关系：num_leaves = 2^(max_depth)。它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。
- max_depth：限制树的深度，和 num_leaves 只需要设置一个。
- min_data_in_leaf（min_child_samples ）：叶子节点中最小的数据量，调大可以防止过拟合。
- min_sum_hessian_in_leaf（min_child_weight）：叶子节点的最小权重和，调大可以防止过拟合。
- bagging_fraction（subsample ）：样本采样比例，同 XGBoost ，调小可以防止过拟合，加快运算速度。
- feature_fraction（colsample_bytree）：列采样比例，同 XGBoost，调小可以防止过拟合，加快运算速度。
- bagging_freq（subsample_freq）：bagging 的频率，0 表示禁止 bagging，正整数表示每隔多少个迭代进行 bagging。
- lambda_l1（reg_alpha）：L1 正则化项，同 XGBoost。
- lambda_l2（reg_lambda）：L2 正则化项，同 XGBoost。
- min_gain_to_split（min_split_gain）：分裂的最小增益阈值。
- drop_rate：Dart 的丢弃率。
- skip_drop：Dart 的跳过丢弃步骤的概率。

最后说一下，

1. 仅仅靠参数的调整和模型的小幅优化，想要让模型的表现有个大幅度提升是不可能的。确实是有一定的提升，但是没有达到质的飞跃。
2. 要想让模型的表现有一个质的飞跃，需要依靠其他的手段，例如特征工程（feature egineering），模型组合（ensemble of model），以及堆叠（stacking）等。