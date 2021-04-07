> https://zhuanlan.zhihu.com/p/89549390

# 深入理解GBDT二分类算法

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig1.png)

## 1. GBDT 分类算法简介

GBDT 无论用于分类还是回归，一直使用的是 CART 回归树。GBDT 不会因为我们所选择的任务是分类任务就选用分类树，这里的核心原因是 GBDT 每轮的训练是在上一轮训练模型的负梯度值基础之上训练的。

这就要求每轮迭代的时候，真实标签减去弱分类器的输出结果是有意义的，即残差是有意义的。如果选用的弱分类器是分类树，类别相减是没有意义的。对于这样的问题，可以采用两种方法来解决：

1. 采用指数损失函数，这样 GBDT 就退化成了 Adaboost，能够解决分类的问题；
2. 使用类似于逻辑回归的 **对数似然损失函数** ，如此可以通过结果的概率值与真实概率值的差距当做残差来拟合；

下面我们就通过二分类问题，去看看 GBDT 究竟是如何做分类的。

## 2. GBDT二分类算法

#### 2.1 逻辑回归的对数损失函数

逻辑回归的预测函数为：
$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}
$$
函数 $h_{\theta}(x)$ 的值有特殊的含义，它表示结果取 $1$ 的概率，因此对于输入 $x$ 分类结果为类别 $1$ 和类别 $0$ 的概率分别为：
$$
\begin{array}{l}
P(Y=1 \mid x ; \theta)=h_{\theta}(x) \\
P(Y=0 \mid x ; \theta)=1-h_{\theta}(x)
\end{array}
$$
下面我们根据上式，推导出逻辑回归的对数损失函数 $L(\theta)$ 。上式综合起来可以写成：
$$
P(Y=y \mid x ; \theta)=\left(h_{\theta}(x)\right)^{y}\left(1-h_{\theta}(x)\right)^{1-y}
$$
然后取似然函数为：
$$
l(\theta)=\prod_{i=1}^{N} P\left(Y=y_{i} \mid x_{i} ; \theta\right)=\prod_{i=1}^{N}\left(h_{\theta}\left(x_{i}\right)\right)^{y_{i}}\left(1-h_{\theta}\left(x_{i}\right)\right)^{1-y_{i}}
$$
因为 $l(\theta)$ 和 $log \ l(\theta)$ 在同一 $\theta$ 处取得极值，因此我们接着取对数似然函数为：
$$
L(\theta)=\sum_{i=1}^{N}\left[y_{i} \log h_{\theta}\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-h_{\theta}\left(x_{i}\right)\right)\right]
$$
最大似然估计就是求使 $L(\theta)$ 取最大值时的  $\theta$ 。这里对 $L(\theta)$ 取相反数，可以使用梯度下降法求解，求得的 $\theta$ 就是要求的最佳参数：
$$
L(\theta)=\sum_{i=1}^{N}\left[y_{i} \log h_{\theta}\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-h_{\theta}\left(x_{i}\right)\right)\right]
$$

#### 2.2 GBDT 二分类原理

逻辑回归单个样本 $(x_i,y_i)$ 的损失函数可以表达为：
$$
L(\theta)=-y_{i} \log \hat{y}_{i}-\left(1-y_{i}\right) \log \left(1-\hat{y}_{i}\right)
$$
其中，$\hat{y_i}=h_{\theta}(x)$ 是逻辑回归预测的结果。假设 GBDT 第 $M$ 步迭代之后当前学习器为 $F(x) = \sum_{m=1}^Mh_m(\theta)$ ，将 $\hat{y_i}$ 替换为 $F(x)$ 带入上式之后，可将损失函数写为：
$$
L\left(y_{i}, F\left(x_{i}\right)\right)=y_{i} \log \left(1+e^{-F\left(x_{i}\right)}\right)+\left(1-y_{i}\right)\left[F\left(x_{i}\right)+\log \left(1+e^{-F\left(x_{i}\right)}\right)\right]
$$
其中，第 $m$ 棵树对应的响应值为（损失函数的负梯度，即伪残差）：
$$
r_{m, i}=-\left|\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}\right|_{F(x)=F_{m-1}(x)}=y_{i}-\frac{1}{1+e^{-F\left(x_{i}\right)}}=y_{i}-\hat{y}_{i}
$$
对于生成的决策树，计算各个叶子节点的最佳残差拟合值为：
$$
c_{m, j}=\underset{c}{\arg \min } \sum_{x_{i} \in R_{m, j}} L\left(y_{i}, F_{m-1}\left(x_{i}\right)+c\right)
$$
由于上式没有闭式解（closed form solution），我们一般使用近似值代替：
$$
c_{m, j}=\frac{\sum_{x_{i} \in R_{m, j}} r_{m, i}}{\sum_{x_{i} \in R_{m, j}}\left(y_{i}-r_{m, i}\right)\left(1-y_{i}+r_{m, i}\right)}
$$
**补充近似值代替过程：**

假设仅有一个样本：
$$
L\left(y_{i}, F(x)\right)=-\left(y_{i} \ln \frac{1}{1+e^{-F(x)}}+\left(1-y_{i}\right) \ln \left(1-\frac{1}{1+e^{-F(x)}}\right)\right)
$$
令
$$
P_{i}=\frac{1}{1+e^{-F(x)}}
$$
则
$$
\frac{\partial P_{i}}{\partial F(x)}=P_{i}\left(1-P_{i}\right)
$$
求一阶导：
$$
\begin{aligned}
\frac{\partial L\left(y_{i}, F(x)\right)}{\partial F(x)} &=\frac{\partial L\left(y_{i}, F(x)\right)}{\partial P_{i}} \cdot \frac{\partial P_{i}}{\partial F(x)} \\
&=-\left(\frac{y_{i}}{P_{i}}-\frac{1-y_{i}}{1-P_{i}}\right) \cdot\left(P_{i} \cdot\left(1-P_{i}\right)\right) \\
&=P_{i}-y_{i}
\end{aligned}
$$
求二阶导：
$$
\begin{aligned}
\frac{\partial^{2} L\left(y_{i}, F(x)\right)}{\partial F(x)^{2}} &=\left(P_{i}-y_{i}\right)^{\prime} \\
&=P_{i}\left(1-P_{i}\right)
\end{aligned}
$$
对于 $L(y_i,F(x) + c)$ 的泰勒二阶展开式：
$$
L\left(y_{i}, F(x)+c\right)=L\left(y_{i}, F(x)\right)+\frac{\partial L\left(y_{i}, F(x)\right)}{\partial F(x)} \cdot c+\frac{1}{2} \frac{\partial^{2} L\left(y_{i}, F(x)\right)}{\partial F(x)^{2}} c^{2}
$$
$L\left(y_{i}, F(x)+c\right)$ 取极值时，上述二阶表达式中的 $c$ 为：
$$
c=-\frac{b}{2 a}=-\frac{\frac{\partial L\left(y_{i}, F(x)\right)}{\partial F(x)}}{2\left(\frac{1}{2} \frac{\partial^{2} L\left(y_{i}, F(x)\right)}{\partial F(x)^{2}}\right)}
$$

$$
=-\frac{\frac{\partial L\left(y_{i}, F(x)\right)}{\partial F(x)}}{\frac{\partial^{2} L\left(y_{i}, F(x)\right)}{\partial F(x)^{2}}}
\stackrel{一阶、二阶导代入}{\Rightarrow}
\frac{y_{i}-P_{i}}{P_{i}\left(1-P_{i}\right)}
$$

$$
\stackrel{r_{i}=y_{i}-P_{i}}{\Rightarrow} \frac{r_{i}}{\left(y_{i}-r_{i}\right)\left(1-y_{i}+r_{i}\right)}
$$

> 注：附泰勒公式求解
>
> ![Taylor1](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Taylor.png)
>
> ![Taylor2](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Taylor2.png)
>
> ![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Taylor3.png)

#### GBDT 二分类算法完整的算法过程如下：

**（1）初始化第一个弱学习器 $F_0(x)$ ：**
$$
F_{0}(x)=\log \frac{P(Y=1 \mid x)}{1-P(Y=1 \mid x)}
$$
其中， $P(Y = 1 | x)$ 是训练样本中 $y=1$ 的比例，利用先验信息来初始化学习器。

**（2）对于建立** $M$ **棵分类回归树** $m=1,2,\cdots,M$ :

**a)** 对 $i=1,2,\cdots,N$ ，计算第 $m$ 棵树对应的响应值（损失函数的负梯度，即伪残差）：
$$
r_{m, i}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F(x)}\right]_{F(x)=F_{m-1}(x)}=y_{i}-\frac{1}{1+e^{-F\left(x_{i}\right)}}
$$
**b)** 对于 $i=1,2,\cdots,N$ ，利用 CART 回归树拟合数据  $r_{m,i}$ ，得到第 $m$ 棵回归树，其对应的叶子节点区域为 $m$ ，其中 $j=1,2,\cdots,J_m$ ，且 $J_m$ 为第  $m$ 棵回归树叶子节点的个数。

**c)** 对于 $J_m$ 个叶子节点区域 $j=1,2,\cdots,J_m$ ，计算出最佳拟合值：
$$
c_{m, j}=\frac{\sum_{x_{i} \in R_{m, j}} r_{m, i}}{\sum_{x_{i} \in R_{m, j}}\left(y_{i}-r_{m, i}\right)\left(1-y_{i}+r_{m, i}\right)}
$$
**d)** 更新强学习器 $F_m(x)$：
$$
F_{m}(x)=F_{m-1}(x)+\sum_{j=1}^{J_{m}} c_{m, j} I\left(x \in R_{m, j}\right)
$$
**(3) 得到最终的强学习器 $F_M(x)$ 的表达式：**
$$
F_{M}(x)=F_{0}(x)+\sum_{m=1}^{M} \sum_{j=1}^{J_{m}} c_{m, j} I\left(x \in R_{m, j}\right)
$$
从以上过程中可知，除了由损失函数引起的负梯度计算和叶子节点的最佳残差拟合值的计算不同，二元 GBDT 分类和 GBDT 回归算法过程基本相似。那二元 GBDT 是如何做分类呢？

将逻辑回归的公式进行整理，我们可以得到 $\log \frac{p}{1-p}=\theta^{T} x$ ，其中 $p=P(Y=1 \mid x)$，也就是将给定输入 $x$ 预测为正样本的概率。逻辑回归用一个线性模型去拟合 $Y = 1|x$ 这个事件的对数几率（odds）$log\frac{p}{1-p}$ 。二元 GBDT 分类算法和逻辑回归思想一样，用一系列的梯度提升树去拟合这个对数几率，其分类模型可以表达为：
$$
P(Y=1 \mid x)=\frac{1}{1+e^{-F_{M}(x)}}
$$

## 3. GBDT 二分类算法实例

#### （1）数据集介绍

训练集如下表所示，一组数据的特征有年龄和体重，把身高大于 1.5 米作为分类边界，身高大于 1.5 米的令标签为 1，身高小于等于 1.5 米的令标签为 0，共有 4 组数据。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Table1.png)

测试数据如下表所示，只有一组数据，年龄为 25 、体重为 65 ，我们用在训练集上训练好的 GBDT 模型预测该组数据的身高是否大于 1.5 米？

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Table2.png)

#### (2) 模型训练阶段

**参数设置：**

- 学习率：learning_rate = 0.1
- 迭代次数：n_trees = 5
- 树的深度：max_depth = 3

**1）初始化弱学习器：**
$$
F_{0}(x)=\log \frac{P(Y=1 \mid x)}{1-P(Y=1 \mid x)}=\log \frac{2}{2}=0
$$
**2）对于建立 $M$ 棵分类回归树** $m=1,2,\cdots,M$：

由于我们设置了迭代次数：n_trees = 5，这就是设置了 M = 5。

**首先计算负梯度**，根据上文损失函数为对数损失时，负梯度（即伪残差、近似残差）为：
$$
r_{m, i}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F(x)}\right]_{F(x)=F_{m-1}(x)}=y_{i}-\frac{1}{1+e^{-F\left(x_{i}\right)}}
$$
我们知道梯度提升类算法，其关键是利用损失函数的负梯度的值作为回归问题提升树算法中的残差的近似值，拟合一个回归树。这里，为了称呼方便，我们把负梯度叫做残差。

现将残差的计算结果列表如下：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Table3.png)

此时将残差作为样本的标签来训练弱学习器 $F_1(x)$ ，即下表数据：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Table4.png)

**接着寻找回归树的最佳划分节点**，遍历每个特征的每个可能取值。从年龄特征值为5 开始，到体重特征为 70 结束，分别计算分裂后两组数据的平方损失（Square Error），$SE_l$ 为左节点的平方损失，  $SE_r$ 为右节点的平方损失，找到使平方损失和 $SE_{sum} = SE_l + SE_r$  最小的那个划分节点，即为最佳划分节点。

例如：以年龄 7 为划分节点，将小于 7 的样本划分为到左节点，大于等于 7 的样本划分为右节点。左节点包括 $x_0$，右节点包括样本 ，$SE_l=0$ , $SE_r = 0.667$, $SE_{sum} = 0.667$ 所有可能的划分情况如下表所示：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Table5.png)

以上划分点的总平方损失最小为 **0.000，**有两个划分点：年龄 21 和体重 60，所以随机选一个作为划分点，这里我们选**年龄 21**。现在我们的第一棵树长这个样子：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig2.png)

我们设置的参数中树的深度 max_depth = 3 ，现在树的深度只有 2，需要再进行一次划分，这次划分要对左右两个节点分别进行划分，但是我们在生成树的时候，设置了三个树继续生长的条件：

- **深度没有到达最大。树的深度设置为 3，意思是需要生长成 3 层。**
- **点样本数 >= min_samples_split**
- **此节点上的样本的标签值不一样（如果值一样说明已经划分得很好了，不需要再分）（本程序满足这个条件，因此树只有2层）**

最终我们的第一棵回归树长下面这个样子： 

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig3.png)

此时我们的树满足了设置，还需要做一件事情，给这棵树的每个叶子节点分别赋一个参数 $c$，来拟合残差。
$$
c_{1, j}=\frac{\sum_{x_{i} \in R_{1, j}} r_{1, i}}{\sum_{x_{i} \in R_{1, j}}\left(y_{i}-r_{1, i}\right)\left(1-y_{i}+r_{1, i}\right)}
$$
根据上述划分结果，为了方便表示，规定从左到右为第 1, 2 个叶子结点，其计算值过程如下：
$$
\begin{array}{ll}
\left(x_{0}, x_{1} \in R_{1,1}\right), & c_{1,1}=-2.0 \\
\left(x_{2}, x_{3} \in R_{1,2}\right), & c_{1,2}=2.0
\end{array}
$$
此时的第一棵树长下面这个样子：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig4.png)

接着更新强学习器，需要用到学习率：learning_rate=0.1，用 $lr$ 表示。更新公式为：
$$
F_{1}(x)=F_{0}(x)+l r * \sum_{j=1}^{2} c_{1, j} I\left(x \in R_{1, j}\right)
$$
为什么要用学习率呢？这是 **Shrinkage** 的思想，如果每次都全部加上拟合值 $c$ ，即学习率为 1，很容易一步学到位导致 GBDT 过拟合。

**重复此步骤，直到** $m>5$  **结束，最后生成 5 棵树。**

下面将展示每棵树最终的结构，这些图都是我 GitHub 上的代码生成的，感兴趣的同学可以去运行一下代码。

[https://github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_GradientBoostingBinaryClassifier](https://github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_GradientBoostingBinaryClassifier)

**第一棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig5.png)

**第二棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig6.png)

**第三棵树：**

![img](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig7.png)

**第四棵树：**

![img](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig8.png)

**第五棵树：**

![img](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/05_GBDT_binary/Fig9.png)

**3）得到最后的强学习器：**
$$
F_{5}(x)=F_{0}(x)+\sum_{m=1}^{5} \sum_{j=1}^{2} c_{m, j} I\left(x \in R_{m, j}\right)
$$

#### （3）模型预测阶段

- $F_0(x) = 0$
- 在 $F_1(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为 **2.0000**。
- 在 $F_2(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为 **1.8187**。
- 在 $F_3(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为 **1.6826**。
- 在 $F_4(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为 **1.5769**。
- 在 $F_5(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为 **1.4927**。

**最终预测结果为：**
$$
\begin{array}{l}
F(x)=0.0000+0.1 *(2.0000+1.8187+1.6826+1.5769+1.4927)=0.8571 \\
P(Y=1 \mid x)=\frac{1}{1+e^{-F(x)}}=\frac{1}{1+e^{-0.8571}}=0.7021
\end{array}
$$

## 4. 手撕GBDT二分类算法

本篇文章所有数据集和代码均在我的GitHub中，地址：

[https://github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning](https://link.zhihu.com/?target=https%3A//github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning)

## 5. GBDT分类任务常见的损失函数

对于 GBDT 分类算法，其损失函数一般有对数损失函数和指数损失函数两种:

（1）如果是指数损失函数，则损失函数表达式为：
$$
L(y, F(x))=\exp (-y F(x))
$$
其负梯度计算和叶子节点的最佳负梯度拟合可以参看 Adaboost 算法过程。

（2）如果是对数损失函数，分为二元分类和多元分类两种，本文主要介绍了 GBDT 二元分类的损失函数。

## 6. 总结

在本文中，我们首先简单介绍了如何把 GBDT 回归算法变成分类算法的思路；然后从逻辑回归的对数损失函数推导出 GBDT 的二分类算法原理；其次不仅用 Python3实现 GBDT 二分类算法，还用 sklearn 实现 GBDT 二分类算法；最后介绍了 GBDT 分类任务中常见的损失函数。GBDT 可以完美的解决二分类任务，那它对多分类任务是否有效呢？如果有效，GBDT 是如何做多分类呢？这些问题都需要我们不停的探索和挖掘 GBDT 的深层原理。让我们期待一下 GBDT 在多分类任务中的表现吧！



--------------------

## QA

1. GBDT 分类算法简介

   > 见 1

2. 逻辑回归的对数似然函数推导

   > 见 2

3. GBDT 二分类原理与推导

   > 见 2

4. GBDT 二分类实例

   > 见 3

5. GBDT 分类任务常见的损失函数

   > 见 5