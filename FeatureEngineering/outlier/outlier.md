异常值指与其他观察值具备显著差异的数据，它们可能是真的异常值也可能是错误。

根据特征的属性（数值或分类），使用不同的方法来研究其分布，进而检测异常值。

## 直方图/箱形图

当特征是数值变量时，使用直方图和箱形图来检测异常值。

下图展示了特征 life_sq 的直方图。

```python
# histogram of life_sq.
df['life_sq'].hist(bins=100)
```

由于数据中可能存在异常值，因此下图中数据高度偏斜。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/1.png)

为了进一步研究特征，我们来看一下箱形图。

```python
# box plot.
df.boxplot(column=['life_sq'])
```

从下图中我们可以看到，异常值是一个大于 7000 的数值。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/2.png)

## 描述统计学



对于数值特征，当异常值过于独特时，箱形图无法显示该值。因此，我们可以查看其描述统计学。

例如，对于特征 life_sq，我们可以看到其最大值是 7478，而上四分位数（数据的第 75 个百分位数据）是 43。因此值 7478 是异常值。

```python
df['life_sq'].describe()
```

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/3.png)

## 条形图

当特征是分类变量时，我们可以使用条形图来了解其类别和分布。

例如，特征 ecology 具备合理的分布。但如果某个类别「other」仅有一个值，则它就是异常值。

```python
# bar chart -  distribution of a categorical
variabledf['ecology'].value_counts().plot.bar()
```

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/4.png)

其他方法：还有很多方法可以找出异常值，如散点图、z 分数和聚类，本文不过多探讨全部方法。