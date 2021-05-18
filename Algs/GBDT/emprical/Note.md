# GBDT处理种类很多的类别特征到底用不用one-hot？

#### Answer1 树分裂角度

「GBDT 对高维稀疏特征的处理效果差」这话在某些情况下是成立的，但更应该去理解为什么 GBDT 对高维稀疏特征的效果差。

设想一个二分类场景，有 1000 个类别特征，其中某个类别特征对应的分类全是 1 或者全是 0，那么 GBDT （其实不光是 GBDT，而是基于决策树的模型都有这个问题）可能会错误学习到一个决策分支：只需要出现某个类别特征，那么分类直接预测为 1 或者 0。在这种情况下，GBDT 可能会产生过拟合，但这本质上是训练数据的问题，应该在数据上下些功夫。

那么使用 label encoding 会有啥问题呢？

比如，一共有 1000 个类别特征，映射到 1-1000 这几个整数上。那么 99 和 100 可能拥有完全不同的意义，但是决策树完全有可能学出来一个决策分支，when label_encoding >= 97 and label_encoding <= 100, then XXX；但这个决策分支实际上是把 99 和 100 当成了有关联的类别来处理的，引入了额外的噪音信息，这个在数据挖掘的过程中更应该避免。

#### Answer2 列抽样角度

不建议变成 one-hot. 个人理解：首先在树模型通常会进行列抽样，如果总共有1000个特征 其他一个类别特征有 500 种类 我们将其 one-hot 在列抽样时其他特征被选中的概率会大大降低 其次 one-hot 不利于之后的特征选择 树模型的特征重要性也会失去一些意义. Label encoding 能解决以上问题 但也有它的缺陷。除这两种编码外，还有 count encoding 和 target encoding等等。实战过程中可以都尝试下，观察metric的好坏。

> 存疑？

#### Answer3 推荐系统角度：还是树分裂

不知道需不需要加的 one-hot 的情况下，我的建议是加。当成数字特征处理，gbdt 树学到的是一段一段的值域划分规则，一般情况下不会有什么问题，但是如果遇到新的未见过类别特征，模型预测可能会存在问题。

以下是实践得出的血泪经验。

我们的推荐应用场景就遇到过这种问题，精排模型，有很多类别特征（算法 id、分类 id 等）。我们一开始也是将类别特征当成数字特征处理的，也实验过 one-hot 处理，但是线上 AB 实验效果相当，所以一直保持当成数字特征处理。

后来遇到 badcase，新上线的召回算法，会给一个新的算法 id，发现新上的算法，初始曝光次数都会特别大，造成总体效果下降。后来分析发现，新上的算法的算法id 落在某个区间段，这个区间段的原算法效果都是比较好的，模型会把新的算法 id当成某一段值域进行处理。后来进行 one-hot 就解决了。

## GBDT 如何处理高维度稀疏特征？

#### Answer1 全面的回答，对比 gbdt 与 lr

高维稀疏特征的时候，lr 的效果会比 gbdt 好，为什么？

- 这个问题我也是思考了好久，在平时的项目中也遇到了不少 case，确实高维稀疏特征的时候，使用 gbdt 很容易过拟合。
- 但是还是不知道为啥，后来深入思考了一下模型的特点，发现了一些有趣的地方。
- 假设有 1w 个样本， y 类别 0 和 1，100 维特征，其中10 个样本都是类别1，而特征 f1 的值为 0，1，且刚好这 10 个样本的 f1 特征值都为 1，其余9990 样本都为 0 (在高维稀疏的情况下这种情况很常见)，我们都知道这种情况在树模型的时候，很容易优化出含一个使用 f1 为分裂节点的树直接将数据划分的很好，但是当测试的时候，却会发现效果很差，因为这个特征只是刚好偶然间跟 y 拟合到了这个规律，这也是我们常说的过拟合。但是当时我还是不太懂为什么线性模型就能对这种 case 处理的好？照理说 线性模型在优化之后不也会产生这样一个式子：y = W1 * f1 + Wi * fi+….，其中 W1 特别大以拟合这十个样本吗，因为反正 f1的值只有 0 和 1，W1 过大对其他 9990 样本不会有任何影响。
- 后来思考后发现原因是因为现在的模型普遍都会带着正则项，而 lr 等线性模型的正则项是对权重的惩罚，也就是 W1 一旦过大，惩罚就会很大，进一步压缩 W1 的值，使他不至于过大，而树模型则不一样，树模型的惩罚项通常为叶子节点数和深度等，而我们都知道，对于上面这种 case，树只需要一个节点就可以完美分割 9990 和 10 个样本，惩罚项极其之小.
- 这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：带正则化的线性模型比较不容易对稀疏特征过拟合。

####  Answer2 维度角度

高纬度的我一般直接用 lr…,在高纬稀疏空间中，往往样本更加线性可分，记得正则化，至于gbdt的回归树我印象中本来就是处理数字变量的。

####  Answer3 FM

对于这种类别特征特别多的情况，可以尝试一下 factorized machines。

## LightGBM/XGBoost需要进行特征选择吗？

决策树模型训练时每次划分会选择信息增益最大的特征进行划分，所以我理解的是训练模型前不需要再手动进行特征选择。

1) 最近在 kaggle 上看到一个帖子提到了用 K-S 检验来选择特征，来减小交叉验证集和测试集得分的差别，也就是**用模型的过拟合程度来进行特征选择**。这算是从另一个角度来考虑特征选择？[Reducing the gap between CV and LB](https://link.zhihu.com/?target=https%3A//www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537)

2) 除此之外，我还遇到一种情况，我做了很多特征，但是感觉可能不同的特征可能包含了重复的信息，这样是否会影响模型的表现呢？

#### Answer1 玄学

特征重复信息的问题，根据我的经验，重复信息基本不影响最终效果。

比如x1是个有效特征，特征重要性占比很高。现在来了个 x7，和x1相关性很大，而且包含更准的信息。那么在新模型里，x1 的特征重要性就会断崖式下跌。模型效果有所提升。x7 基本取代了之前 x1 的地位。

#### Answer2 正则项

我的简单结论是，对于有惩罚项的机器学习算法，不需要。

特征选择更多见到的是逻辑回归这种有考虑置信区间的算法，因为要考虑置信区间（达不到95%以上的可信度我就不要了），所以要做特征取舍，因为要做特征取舍，所以要考虑多重共线性问题，因为共线性会直接影响特征的重要性表现。

机器学习不考虑置信区间（有些结构你也没法考虑），只考虑如何对于现有的数据集用一个 $f(x)$ 去更好的拟合 $y$，只要加入的特征没有造成数据集内的过拟合就都纳入进来，通过惩罚项来进行一刀切。

所以说，目的不一样，机器学习只要求拟合好本数据集，也强大在拟合能力上，泛化的表现取决于数据集的采样情况，很适合大数据场景下，而数据较少的情况，从泛化角度考虑（看你是为了竞赛还是为了实际使用）适合传统方法。

## 噪声、共线性、太多的特征对gbdt的负面影响

说说最常见的，共线性问题，特征相关性太高引发的问题：

**1、特征重要性的有效性变差**，比如某个强特本来特征重要性是很高的，但是因为存在多组相关性很强的强特，极端的例子，直接赋值几列取值完全相同的特征列，则在feature importance中，这个强特的特征重要性将被稀释；具体测试案例可见：

[https://datascience.stackexchange.com/questions/12554/does-xgboost-handle-multicollinearity-by-itself](https://datascience.stackexchange.com/questions/12554/does-xgboost-handle-multicollinearity-by-itself)

> There is an answer from Tianqi Chen (2018).
>
> This difference has an impact on a corner case in feature importance analysis: the correlated features. Imagine two features perfectly correlated, feature A and feature B. For one specific tree, if the algorithm needs one of them, it will choose randomly (true in both boosting and Random Forests).
>
> However, in Random Forests™ this random choice will be done for each tree, because each tree is independent from the others. Therefore, approximatively, depending of your parameters, 50% of the trees will choose feature A and the other 50% will choose feature B. So the importance of the information contained in A and B (which is the same, because they are perfectly correlated) is diluted in A and B. So you won’t easily know this information is important to predict what you want to predict! It is even worse when you have 10 correlated features…
>
> In boosting, when a specific link between feature and outcome have been learned by the algorithm, it will try to not refocus on it (in theory it is what happens, the reality is not always that simple). Therefore, all the importance will be on feature A or on feature B (but not both). You will know that one feature has an important role in the link between the observations and the label. It is still up to you to search for the correlated features to the one detected as important if you need to know all of them.

To summarise, Xgboost does not randomly use the correlated features in each tree, which random forest model suffers from such a situation.

**2、共线性会影响模型的泛化性能**

https://www.kaggle.com/c/quora-question-pairs/discussion/33876

正如这个作者所说，太多共线性特征的存在确实会使得模型的泛化性能下降，当然，这里指的是“过多”，如果是很少量共线性特征存在其实对于模型影响很轻微，但是如果存在大量共线性特征，尤其是比赛的时候暴力的特征衍生，会产生相当多相关性很高的特征，从而导致 gbdt 在训练的过程中重复采样相关性很高的特征，使得模型的效果变差，具体可见 kaggle_ieee 的 kris 分享的方案，通过删除大量冗余的 V 特征，local cv 上升了千五，b榜上涨千4（事后分别测试的）。

补充：针对不少人的提问，这里补充一下吧，**xgb 或 lgb 的泛化性能好的重要原因之一（注意是之一，下面所说的只是其中的一方面）可以通过行列采样构建不同的特征空间和样本空间下的基模型，集成学习的核心“好而不同”，xgb 或 lgb 解决了“不同”的问题，至于单树的“好”或者“坏”在 gbm 的框架下影响并不大，单棵树的表达能力不够，多点树来凑，具体的应用中的例子就是，对于过拟合问题，设置大一点的行列采样比例则过拟合的程度很快就能降下来，可以行列采样和最大深度对于gbdt整体模型的影响是比较显著的，立竿见影。 ** 而之所以共线性会影响泛化性能，前提存在大量的共线性，比如有 10 个特征 A、B。。。。J，假设这10个特征完全独立，并且假设我们每次列采样比例为0.2，也就是每次采样2个特征出来训练一个基模型，则一共有 10*9/2=45 种组合，这个时候，我们再假设一个极端的情况，B~J的特征完全一样，是相同特征的复制，这个时候，不管我们怎么采样永远只有两种特征的组合，即 A 和（B~J）其中任何一个特征的组合都是一样的或者（B~J）内部的特征组合，显然，基模型的可以使用的子空间的种类大大降低，集成模型的所谓“不同”大大弱化了，多样性降低，自然模型的泛化性能下来了。

**3、不重要的（噪声）特征是否需要删除？gbdt算法的分裂的过程中可以自动做到特征选择那么是否意味着我可以无限做特征衍生，反正模型会自动筛选其中的好坏特征？**

首先，暴力衍生不太可取，一方面太耗时间和内存，一方面容易产生高相关性（或者高局部相关的）特征与噪声，高相关性特征可以通过相关系数来剔除，但是高局部相关特征的检验很复杂并且麻烦，而噪声特征的处理也比较繁琐耗时。

> 噪声变量不会带来大的增益，所以实际上树在分裂的时候不会考虑他们从而也不会影响树的分裂，因此无限大的特征维度对 tree 没有影响

回答：这仅对于非常巨大的的数据集是正确的，因为大量的训练集中的样本数量可以很好地涵盖所有特征空间由于引入新特征而产生的变化。但是在实践中，样本数量往往有限，如果特征维数足够大（比如暴力的特征衍生），最终会带来很多采样噪声，因为数据越多维数，对可能的分布空间的覆盖范围的能力就越弱。

**最终与标签偶然相关联的弱变量上的噪声可能会限制增强算法的有效性**，而这在决策树中的更深的深度的拆分中更容易发生，在该决策树中，已评估的数据已被分组为一个小的子集。

**添加的变量越多，弱相关变量就越有可能恰好适合某些特定组合的分割选择**，然后创建新的分支，但是这是噪声产生的模型分支对于未来的预测有很大偏误。

实际上，我发现XGBoost在小范围内对噪声非常鲁棒。但是，我也发现，出于类似的原因，它有时会选择质量较差的特征变量，而不是关联性更好的数据。**因此，这不是“变量越多对XGBoost越好”的算法，您确实需要考虑可能的低质量特征。**

1. 更多内存占用，更慢的装载速度，更慢的处理速度等。
2. 每个特征需要最少数量的样本才能有效学习模型。通过包含冗余特征，此样本：特征比率得以减小，这使许多模型更难确定特征是否是有用的预测因子。（补充一下，这里作者的意思是这样的，就是假设存在过多的垃圾特征，那么树在分裂的过程中难免会用到这些垃圾特征导致树的分裂，继而导致每个新的分支的样本数量的减少，那么比如我们设置叶子节点最小样本数量、最大叶子节点数量等超参数的时候就很容易导致树因为达到了预设条件而提前终止分裂，退一步说，即使没有超参数的限制，因为被垃圾特征“抢走”了不少分裂的样本，可能导致真正有效的特征最终只能在少数样本上分裂，这将导致模型无法正确评估这类实际上有效的特征并且也无法正确利用有效的特征）。

另外补充一点，关于噪声特征，我们可以使用 kris 验证法的变体来测试，大体上分为两种，一种是完全的废物特征，在训练集和验证集上的得分都很低，这类特征少量存在其实 gbdt 是可以很好的 handle的，但是当这类特征大量存在的时候非常影响模型的性能，比如我有两个强特和 2000 个垃圾特征，一起训练和剔除垃圾特征之后训练的结果差异性会相当之大的；一种是“伪装的强特”，在训练集上得分很高而在测试集上得分很低，这类特征仅仅有少数就能够影响模型的泛化性能了。

关于kris验证方法的变体大概形式如下：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/emprical/Fig1.png)

实际上就是每个特征单独跑个交叉验证，参数尽量宽松设置以最大化特征的“潜能”，通过交叉验证的平均训练集与验证集得分来判定特征属于哪一种特征。这种方法的好处在于可以很直观而独立的了解每一个特征的偏移情况，不足之处在于无法确定哪一些“特征组合”的偏移情况，因为我们需要知道的是，关于特征偏移，实际上很多时候都是和特征的组合相关的而不是单特征相关（当然偏移很厉害的单特征也是存在这里只是提出一种更常见的现象），比如上面这些特征，单个特征的偏移程度可能都不会非常大，最大的大概就 5 个点auc，但是我们将两个偏移 3 个点的特征一起入模计算就会发现偏移可能会到达 9~10个点，因此针对这类现象，我们可以对偏移严重的特征进行组合的穷举然后两个两个特征入模去计算从而得出哪些特征组合的偏移严重，不过很多时候单个特征的问题能够解决，高阶特征组合的问题自然而然也能得到解决，后续会把前段时间磕磕绊绊的ieee的总结结果搬上来，感觉多看看大佬的打比赛的思路还是很有收获的。



>  作者您好，我有几个问题，第一，特征重要性衡量出来的高相关特征，意思是特征重要性给分给的高的特征对吗?那既然给分高了，删掉了，不会影响模型性能吗？如果删掉了反而导致模型性能变好，那特征重要性岂不是变成了一个特征冗余性？加入高相关性特征会导致模型变差因为有共线性这样表述是不妥的，因为机器学习建模依赖的就是目标和特征之间的相关性。我觉得中文应该用特征相似性这里比较准确。还有就是这里使用的相关系数，是哪一种相关系数?斯皮尔曼相关系数，皮尔逊相关系数，最大互信息率相关系数，等等，不确定的是相关系数衡量出来的线性关系，非线性关系，或者手机复合型关系，是不是真的就是特征和特征之间那种相关性，这样剔除，可能还要考虑和分析。
>
> 第一个问题，举个例子，假设你把标签直接作为特征加入原始数据则这个特征的重要性会非常高而模型的泛化性能会非常差，这里说的是 feature importance 的缺陷，确实对于冗余特征无法充分反应；这里的高相关说的是特征之间的相关性而不是特征和标签之间；关于相关系数，常用的是spearman和pearson，其它的基本么有使用过，这里所说的相关性指的是相关系数衡量的相似性； 实际数据的测试中，我是使用多组相关系数阈值进行测试，一般情况下能保证模型至少不变差的情况下降低特征的数量。

## xgboost，特征多好还是少好？

#### Answer1 降维角度，快手

1000个特征太多啦，可能出现维度灾难的情况。从实践经验来看，相比于丢掉一些原始特征，更优的策略是做降维。你可以尝试不同的降维方法与不同维度保留个数的组合，用网络搜索（GridSearch）来确定最优的降维方法。

#### Answer2 数据分析角度

对于这种表格型的数据，模型的效果和特征的数量我觉得没有直接的关系。我由于打的比赛比较多，所以经常会发现Top的队伍可能只用了几十个特征，他的线上效果却比我成百上千的特征要好上不少。

做特征要看从什么角度来出发，如果只是盲目的堆砌一大推特征可能还不如基于对数据的分析，对业务的理解再抽象成个位数的特征来的效果好。

不过数据新手的确会经常遇到这样的问题，在这里我的建议去找一些和你场景相类似的比赛里面Top团队所贡献的开源方案，通过他们的方案去学习总结然后形成自己的一套特征工程体系。

最后我想说的是，特征不在于多少，而在于精，在于你是否对这个场景有足够的理解。

## XGBoost中无序类别特征需要one-hot吗？

#### Answer1 

你可以实验一下，用不用onehot的区别。

我的经验是类别稍多，>4的时候，onehot 效果就会很差。甚至你把类别随机排序转成整数，把离散特征当连续特征来处理这种不科学的做法也比 onehot 强。

根据我的经验，最好的处理离散特征的方法是 lightgbm 内置的处理离散特征方法。具体做法是将离散特征转为整数，然后在 dataset 方法标注离散特征有哪些。xgboost 应该也支持这个功能。

大概原理是将分裂离散特征时，n 个类分裂成 n 个节点，按节点值排序，然后尝试不同的节点值切分点。多类 vs 多类的模式。

或者稍差一点，把类别转化为该类正样本率，维护一个类别到正样本率的字典

## 为什么xgboost不适合高维稀疏特征？

#### Answer1

跟GBDT一样，做结点分裂的时候使用贪心算法找特征的最佳切分点。

如果对数据进行one-hot编码，那每个特征就只能按0-1切分了。本来别人是想切哪里切哪里，怎么切目标函数低怎么切，结果one-hot一下相当于把能切的方案都给限制了。同样的道理，基于回归树的这一类方法也不适合对数据做离散化处理。

> 讲道理 rf 也是贪心的寻找分割点，但是实际中，rf 对于稀疏数据的效果远好于 boosting 的方法，除了随机性防止过拟合外，还有别的原因吗?



> 不同意你的观点。
>
> [Factors · Issue #95 · dmlc/xgboost](http://link.zhihu.com/?target=https%3A//github.com/dmlc/xgboost/issues/95)
>
> [comment:re sklearn -- integer encoding vs 1-hot (py) · Issue #1 · szilard/benchm-ml](http://link.zhihu.com/?target=https%3A//github.com/szilard/benchm-ml/issues/1)
>
> 这两个issue中，陈天奇都提到可以one-hot encoding
>
> 没有说不能做one hot呀（在说这句话之前我已经忘了回答的什么了，我记得我想表达的意思是“不适合做one hot”）..按陈天奇的意思应该是说xgboost“能处理”怎样的数据吧，树结构当然是能处理离散变量的数据啊，但是他也明显提到了 Normal tree growing algorithm only support dense numerical features, and have to support one-hot encoding factor explicitly for computation efficiency reason. 支持 one hot是为了提高计算效率。举个最明显的例子，日期数据，做了one hot 再加上使用列采样会发生什么？如果不做 one hot，采样时选中了日期特征，那结点就可能会以全量的日期数据来判断如何分裂；如果做了one hot，采样时选中了一部分日期特征，那么只可能以一部分（且一般是极小部分的日期数据）来做结点分裂，显然，这在绝大多数情况下训练出来的模型都没有不做 one hot好，加速效果倒是实打实的。后面有个老哥也有提到关于lightGBM的离散数据问题。

#### Answer2 效率

高维稀疏的ID类特征会使树模型的训练效率变得极为低效，且容易过拟合。

树模型训练过程是一个贪婪选择特征的算法，要从候选特征集合中选择一个使分裂后信息增益最大的特征来分裂。按照高维的ID特征做分裂时，子树数量非常多，计算量会非常大，训练会非常慢。

同时，按 ID 分裂得到的子树的泛化性也比较弱，由于只包含了对应 ID 值的样本，样本稀疏时也很容易过拟合。

## 用lightgbm建模时，标签编码后的类别型特征还需要归一化吗？还是只需要对连续型特征做归一化呢？

比如星期转换成1-7以后还需要归一化吗？

#### Answer1

1.树模型对于特征量纲没有要求，连续特征，类别型特征都不要归一化操作，
  一是因为树模型不是利用SGD等优化算法进行优化；
  二是LightGBM中回归树生长过程中，是利用特征的直方图寻找最优的特征，以及分裂点，因此这个过程只关心取值的顺序，即使归一化之后，各个样本的取值的顺序依然不会改变，所以没有必要；

2.对于类别型的特征，传统的机器学习模型是需要先利用one-hot编码，而在LightGBM中只需要提前将类别映射到非负整数即可(`integer-encoded categorical features`)，例如，进行如下编码mapping`{'川建国': 1, '傻蛋': 2, '其他': 0}`，在官方文档中也建议使用从0开始的连续的数值进行编码，当训练集中的某个类别型的特征取值个数超大，可以将其看做是连续特征看待，或者进行embedding编码。因此你说的星期只需要编码为1~7即可

#### Answer2 官方文档

> It is common to represent categorical features with one-hot **encoding**, but this approach is suboptimal for tree **learners**. Particularly for high-**cardinality** categorical features, a tree built on one-hot features tends to be **unbalanced** and needs to grow very deep to achieve good accuracy.
> Instead of one-hot **encoding**, the optimal solution is to split on a categorical feature by partitioning its categories into 2 subsets. If the feature has `k` categories, there are `2^(k-1)-1` possible partitions. But there is an efficient solution for regression trees[[8\]](https://link.zhihu.com/?target=https%3A//lightgbm.readthedocs.io/en/latest/Features.html%23references). It needs about `O(k*log(k))` to find the optimal partition.
> The basic idea is to sort the categories according to the training objective at each split. More specifically, LightGBM sorts the **histogram**(for a categorical feature) according to its **accumulated** values (`sum_gradient/sum_hessian`) and then finds the best split on the sorted **histogram**.

## Xgboost原接口训练时，为什么必须要传一个evals验证集？

#### Answer1

训练过程中引入 eval set 是为了做 early stopping。理论上 xgboost 模型中树的棵树可以无限增长下去，直到最后损失函数在训练集上为 0。但这时候模型复杂度非常高，很容易过拟合。所以训练过程中加入 eval set，如果出现 loss 在训练集上下降但在验证集上上升，说明出现过拟合迹象了，可以提前终止模型训练。

另外 eval set 参数不是必填的，可以自行指定树最多有多少棵来强制终止训练。

至于为什么有些模型训练时不用 eval set，这类模型复杂度不会随训练过程增长，比如线性回归之类的。这时一般训练过程中发生的现象是，训练集上的 loss 会先降再升，这时候一般用训练集上 loss 极小化作为终止条件，可以不引入 eval set。

#### Answer2

验证数据集在训练过程中本身不起作用，但是用来时时观测你的模型是不是过拟合，已经性能变化。

#### Answer3

验证集的作用是同步检测模型是否过拟合，原理上可以选择不用，也可以从 train 里面随机抽取出来，目前很多程序就写死了需要提供 evals，如 TF

