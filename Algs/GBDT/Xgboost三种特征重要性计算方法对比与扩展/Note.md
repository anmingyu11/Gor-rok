> https://zhuanlan.zhihu.com/p/355884348

# Xgboost 三种特征重要性计算方法对比与扩展

## 特征重要性

### 作用与来源

特征重要性，我们一般用来观察不同特征的贡献度。排名靠前的，我们自然而然的认为，它是重要的。

这一思路，通常被用来做**特征筛选**。剔除贡献度不高的尾部特征，增强模型的鲁棒性的同时，起到特征降维的作用。

另一个方面，则是用来做**模型的可解释性**。**我们期望的结果是：重要的特征是符合业务直觉的；符合业务直觉的特征排名靠前。**

在实际操作中，我们一般用树模型的分类节点做文章。常用的就是 XGB 和其他一般树模型。

### XGB 遇到的问题

XGB 很方便，不仅是比赛的大杀器，甚至贴心的内置了重要性函数。但在实际使用过程中，常常陷入迷思。

有如下几个点的顾虑：

1. 这些特征重要性是如何计算得到的？
2. 为什么特征重要性不同？
3. 什么情况下采用何种特征重要性合适？

今天我们就借这篇文章梳理一下。

XGB 中常用的三种特征重要性计算方法，以及它的使用场景。除此之外，再看两个第三方的特征重要性计算方法，跳出内置函数，思考其中的差异。

最后回到类似的树模型特征计算方法，进行特征重要性的一般方法总结。

以下场景非特殊说明，均针对 python 包体下的 xgb 和sklearn。

## XGB 内置的三种特征重要性计算方法

### weight

`xgb.plot_importance` 这是我们常用的绘制特征重要性的函数方法。其背后用到的贡献度计算方法为`weight`。

- ‘weight’ - the number of times a feature is used to split the data across all trees.

简单来说，就是在子树模型分裂时，用到的特征次数。这里计算的是所有的树。这个指标在 R 包里也被称为**`frequency`**。

### gain

`model.feature_importances_` 这是我们调用特征重要性数值时，用到的默认函数方法。其背后用到的贡献度计算方法为`gain`。

- ‘gain’ - the average gain across all splits the feature is used in.

gain 是信息增益的泛化概念。这里是指，节点分裂时，该特征带来信息增益（目标函数）优化的平均值。

### cover

`model = XGBRFClassifier(importance_type = 'cover')` 这个计算方法，需要在定义模型时定义。之后再调用`model.feature_importances_` 得到的便是基于`cover`得到的贡献度。

- ‘cover’ - the average coverage across all splits the feature is used in.

cover 形象来说，就是树模型在分裂时，特征下的叶子结点涵盖的样本数除以特征用来分裂的次数。分裂越靠近根部，cover 值越大。

## 使用场景

weight 将给予数值特征更高的值，因为它的变数越多，树分裂时可切割的空间越大。所以这个指标，会掩盖掉重要的枚举特征。

gain 用到了熵增的概念，它可以方便的找出最直接的特征。即如果某个特征的下的0，在 label 下全是 0 ，则这个特征一定会排得靠前。

cover 对于枚举特征，会更友好。同时，它也没有过度拟合目标函数，不受目标函数的量纲影响。

调用它们的方式如下[3](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A3)

```python3
# Available importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
f = 'gain'
XGBClassifier.get_booster().get_score(importance_type= f)
```

举个例子，我们来做西瓜分类任务。西瓜有颜色、重量、声音、品种、产地等各类特征。其中，生活（业务）经验告诉我们，声音响的瓜甜。我们构建了这么一个模型，判断西瓜甜不甜。输出三类特征重要性。这时会看到一些矛盾的现象。

在 `weight` 解释下，声音这类枚举值特征并不会很靠前。反而是重量这类连续特征会靠前。为什么会这样子，是因为连续特征提供了**更多的切分状态空间**，这样的结果，势必导致它在树模型的分裂点上占据多个位置。与此同时，**重要的枚举值特征靠后**了。而且我们还能预见的是，同一个量纲下，上下界越大的特征，更有可能靠前。

如何解决这个矛盾点，那就是采用 `gain` 或者 `cover` 方法。因为声音这个特征，必然能带来更多的信息增益，减少系统的熵，所以它在信息增益数值上，一定是一个大值。在树模型的分类节点上，也一定是优先作为分裂点，靠近根部的。

在实践中，也会发现，`gain` 排出来的顺序的**头尾部值差距较大**，这是因为信息增益计算时，后续的优化可能都不是一个量级。类似于神经网络在优化损失函数时，后续的量纲可能是十倍、百倍的差异。所以，综上而言，如果**有下游业务方**，更建议用 `cover` 的特征重要性计算方法。当然，如果是单纯的模型调优，`gain` 能指出最重要的特征。这些特征，某些场景下还能总结成硬规则。

## 其他重要性计算方法

除了上述内置的重要性计算方法外，还有其他第三方计算方式。

这里介绍两种，一个是 [permutation importance](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/permutation_importance.html%23permutation-importance)[5](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A5)[6](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A6)，另一个是 shap[7](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A7)。

### permutation

Permutation 的逻辑 [8](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A8) 是：如果这个特征很重要，那么我们打散所有样本中的该特征，则最后的优化目标将折损。这里的折损程度，就是特征的重要程度。

由于其计算依赖单一特征，所以对非线形模型更友好。同时，如果特征中存在多重共线性，共线的特征重要性都将非常靠后。这是因为混淆单一特征，不影响另一个特征的贡献。这样的结果是，即使特征很重要，也会排得很靠后。

针对多重共线特征，[sklearn](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html%23handling-multicollinear-features) 文档中提到了一种解决方法[9](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A9)：计算特征间的 spearman rank-order correlations，取得每一组中的头部特征，再进行特征重要性计算。这种方法，实际上是在解决特征共线的问题。

### shap

另一个特征重要性计算方法 shap，在之前的文章：[机器学习模型的两种解释方法](https://link.zhihu.com/?target=https%3A//kuhungio.me/interpretable-machine-learning.md)[10](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A10)有过介绍。其计算方法，利用了博弈论的知识，即特征的边际递减效应。通俗来说，就是轮流去掉每一个特征，算出剩下特征的贡献情况，以此来推导出被去除特征的边际贡献。该方法是目前唯一的逻辑严密的特征解释方法（但实际使用体感一般，详情可参见上篇文章）。

## 其他模型的特征重要性计算方法

对于同样的树模型，random forest 和 gbdt，同样也有内置的特征重要性。

randomforest 使用 `rf.feature_importances_` 得到特征重要性[11](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A11)。其中，分类任务计算的是 gini 不纯度/信息熵。回归任务计算的是树的方差。

这种基于**不纯度**（Mean Decrease in Impurity）的方法，实际上会有两个问题存在：一：会给予变量空间更大的特征更多关注，而二分类特征则会靠后。二：结果的拟合是基于训练集的，存在过拟合风险，没有验证集数据做验证。

针对上述的问题，建议通过 out-of-bag（OOB）方法[12](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A12)[13](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A13)，在剩下数据上做验证，结合 permutation[14](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A14) 计算特征重要性。

此外，gbdt 也是基于**不纯度**计算的特征重要性，不过其在单棵树上，是回归树[15](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A15)，所以不是基于gini系数，而是 MSE 或 MAE

至于为什么它们同为树模型，且都是基于不纯度计算的重要性，但结果不同。主要有两个，一个是它们的树结构不同；第二个则是，它们的优化对象不同。

## 特征重要性小结

总结起来，我们可以可以发现，这里特征重要性的计算，其实分为了两类。一类是**基于优化过程**的，如树模型都在采用的不纯度。另一类则是**基于外部结果**，通过调整或引入变量，观察对目标的影响。

目前的趋势是，不纯度的局限被大家认识到，越来越多地采用外部结果的方式，来保证特征重要性的一致性和稳定性[16](https://link.zhihu.com/?target=https%3A//kuhungio.me/2021/three-feature-importances-in-xgb/%23fn%3A16)。

对于不稳定性，可以观察到的一个现象是，新特征的加入，很可能打乱（而不是插入）头部特征的排名。如何减缓这个现象？打乱，其实说明了两个问题。一：特征重要性排名的工具有问题，不具有鲁棒性，应当采用多种特征重要性工具对比。二：特征间存在相关性，树模型在解释时，也应当进行相关性剔除。

三种直观的外部检验方法：

1. 将特征随机打散，其实就是 permutation 的计算方式。
2. 引入随机变量列，观察其他特征相对于它的位置排序。
3. 去掉特征，观察目标函数的升降情况。

