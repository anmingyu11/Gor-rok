## 不必要数据类型 1：信息不足/重复

有时一个特征不提供信息，是因为它拥有太多具备相同值的行。

如何找出重复数据？

我们可以为具备高比例相同值的特征创建一个列表。

例如，下图展示了 95% 的行是相同值的特征。

```python
num_rows = len(df.index)low_information_cols = [] #
for col in df.columns:    
  cnts = df[col].value_counts(dropna=False)    
  top_pct = (cnts/num_rows).iloc[0]        
  if top_pct > 0.95:        
    low_information_cols.append(col)        
    print('{0}: {1:.5f}%'.format(col, top_pct*100))
    print(cnts)
    print()
```

我们可以逐一查看这些变量，确认它们是否提供有用信息。（此处不再详述。）

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/5.png)

如何处理重复数据？

我们需要了解重复特征背后的原因。当它们的确无法提供有用信息时，我们就可以丢弃它。

## 不必要数据类型 2：不相关

再次强调，数据需要为项目提供有价值的信息。如果特征与项目试图解决的问题无关，则这些特征是不相关数据。

如何找出不相关数据？

浏览特征，找出不相关的数据。

例如，记录多伦多气温的特征无法为俄罗斯房价预测项目提供任何有用信息。

如何处理不相关数据？

当这些特征无法服务于项目目标时，删除之。

## 不必要数据类型 3：复制

复制数据即，观察值存在副本。

复制数据有两个主要类型。

#### 复制数据类型 1：基于所有特征

如何找出基于所有特征的复制数据？

这种复制发生在观察值内所有特征的值均相同的情况下，很容易找出。

我们需要先删除数据集中的唯一标识符 id，然后删除复制数据得到数据集 `df_dedupped`。对比 `df` 和 `df_dedupped` 这两个数据集的形态，找出复制行的数量。

```python
# we know that column 'id' is unique, but what if we drop it
df_dedupped = df.drop('id', axis=1).drop_duplicates()
# there were duplicate rows
print(df.shape)
print(df_dedupped.shape)
```

我们发现，有 10 行是完全复制的观察值。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/6.png)

如何处理基于所有特征的复制数据？

**删除这些复制数据。**

#### 复制数据类型 2：基于关键特征

如何找出基于关键特征的复制数据？

有时候，最好的方法是删除基于一组唯一标识符的复制数据。

例如，相同使用面积、相同价格、相同建造年限的两次房产交易同时发生的概率接近零。

我们可以设置一组关键特征作为唯一标识符，比如 `timestamp`、`full_sq`、`life_sq`、`floor`、`build_year`、`num_room`、`price_doc`。然后基于这些特征检查是否存在复制数据。

```python
key = ['timestamp', 'full_sq', 'life_sq', 'floor','build_year', 'num_room', 'price_doc']
df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)
```

基于这组关键特征，我们找到了 16 条复制数据。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/7.png)

如何处理基于关键特征的复制数据？

删除这些复制数据。

```python
# drop duplicates based on an subset of variables.
key = ['timestamp', 'full_sq', 'life_sq', 'floor','build_year', 'num_room', 'price_doc']
df_dedupped2 = df.drop_duplicates(subset=key)
print(df.shape)
print(df_dedupped2.shape)
```

删除 16 条复制数据，得到新数据集 `df_dedupped2`。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/outlier/8.png)