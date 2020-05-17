# 不一致数据

在拟合模型时，数据集遵循特定标准也是很重要的一点。我们需要使用不同方式来探索数据，找出不一致数据。大部分情况下，这取决于观察和经验。不存在运行和修复不一致数据的既定代码。

下文介绍了四种不一致数据类型。

## 不一致数据类型 1：大写

在类别值中混用大小写是一种常见的错误。这可能带来一些问题，因为 Python 分析对大小写很敏感。

如何找出大小写不一致的数据？

我们来看特征 sub_area。

```python
df['sub_area'].value_counts(dropna=False)
```

它存储了不同地区的名称，看起来非常标准化。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/1.png)

但是，有时候相同特征内存在不一致的大小写使用情况。「Poselenie Sosenskoe」和「pOseleNie sosenskeo」指的是相同的地区。

如何处理大小写不一致的数据？

为了避免这个问题，我们可以将所有字母设置为小写（或大写）。

```python
# make everything lower case.
df['sub_area_lower'] = df['sub_area'].str.lower()df['sub_area_lower'].value_counts(dropna=False)
```

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/2.png)

## 不一致数据类型 2：格式

我们需要执行的另一个标准化是数据格式。比如将特征从字符串格式转换为 DateTime 格式。

如何找出格式不一致的数据？

特征 timestamp 在表示日期时是字符串格式。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/3.png)

如何处理格式不一致的数据？

使用以下代码进行格式转换，并提取日期或时间值。然后，我们就可以很容易地用年或月的方式分析交易量数据。

```python
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')

df['year'] = df['timestamp_dt'].dt.year
df['month'] = df['timestamp_dt'].dt.month
df['weekday'] = df['timestamp_dt'].dt.weekday

print(df['year'].value_counts(dropna=False))
print()
print(df['month'].value_counts(dropna=False))
```

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/4.png)

## 不一致数据类型 3：类别值

分类特征的值数量有限。有时由于拼写错误等原因可能出现其他值。

如何找出类别值不一致的数据？

我们需要观察特征来找出类别值不一致的情况。举例来说：

由于本文使用的房地产数据集不存在这类问题，因此我们创建了一个新的数据集。例如，city 的值被错误输入为「torontoo」和「tronto」，其实二者均表示「toronto」（正确值）。

**识别它们的一种简单方式是模糊逻辑（或编辑距离）。该方法可以衡量使一个值匹配另一个值需要更改的字母数量（距离）。**

已知这些类别应仅有四个值：「toronto」、「vancouver」、「montreal」和「calgary」。计算所有值与单词「toronto」（和「vancouver」）之间的距离，我们可以看到疑似拼写错误的值与正确值之间的距离较小，因为它们只有几个字母不同。

```python
from nltk.metrics import edit_distance

df_city_ex = pd.DataFrame(data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})

df_city_ex['city_distance_toronto'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'toronto'))

df_city_ex['city_distance_vancouver'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'vancouver'))
df_city_ex
```

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/5.png)

如何处理类别值不一致的数据？

我们可以设置标准将这些拼写错误转换为正确值。例如，下列代码规定所有值与「toronto」的距离在 2 个字母以内。

```python
msk = df_city_ex['city_distance_toronto'] <= 2
df_city_ex.loc[msk, 'city'] = 'toronto'

msk = df_city_ex['city_distance_vancouver'] <= 2
df_city_ex.loc[msk, 'city'] = 'vancouver'

df_city_ex
```

![img](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW85zIjvl3dHYr5Ak4xRT7yO78BLCa79FshHibvicRgwHsVTcNcsqV1wXfflJ2hjgj86lF6NFIRL1EDg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

## 不一致数据类型 4：地址

地址特征对很多人来说是老大难问题。因为人们往数据库中输入数据时通常不会遵循标准格式。

如何找出地址不一致的数据？

用浏览的方式可以找出混乱的地址数据。即便有时我们看不出什么问题，也可以运行代码执行标准化。

出于隐私原因，本文采用的房地产数据集没有地址列。因此我们创建具备地址特征的新数据集 df_add_ex。

```python
# no address column in the housing dataset. So create one to show the code.
df_add_ex = pd.DataFrame(['123 MAIN St Apartment 15', '123 Main Street Apt 12   ', '543 FirSt Av', '  876 FIRst Ave.'],columns=['address'])
df_add_ex
```

我们可以看到，地址特征非常混乱。

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/6.png)

如何处理地址不一致的数据？

运行以下代码将所有字母转为小写，删除空格，删除句号，并将措辞标准化。

```python
df_add_ex['address_std'] = df_add_ex['address'].str.lower()df_add_ex['address_std'] = df_add_ex['address_std'].str.strip() # remove leading and trailing whitespace.
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\.', '') 
# remove period.
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bstreet\\b', 'st') 
# replace street with st.
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bapartment\\b', 'apt') # replace apartment with apt.
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bav\\b', 'ave') 
# replace apartment with apt.
df_add_ex
```

现在看起来好多了：

![img](/Users/helloword/Anmingyu/Gor-rok/FeatureEngineering/inconsistence/7.png)

结束了！我们走过了长长的数据清洗旅程。

现在你可以运用本文介绍的方法清洗所有阻碍你拟合模型的「脏」数据了。

