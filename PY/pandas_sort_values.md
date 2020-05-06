> https://zhuanlan.zhihu.com/p/35013079

在之前的文章里我们详细的讨论了`pandas`中的[pandas.loc](https://zhuanlan.zhihu.com/p/35012436)方法以及[pandas.iloc](https://zhuanlan.zhihu.com/p/35012884)方法，今天在看[《Python数据分析实战》](https://link.zhihu.com/?target=https%3A//book.douban.com/subject/26854249/)的时候又发现了一个`pandas`中已经被*deprecated*的方法，我们今天就来聊一聊`pandas.DataFrame.sort_values`方法，同样的我们先看[文档](https://link.zhihu.com/?target=http%3A//pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html)：

```python
DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
```

> Sort by the values along either axis

可以看到这个方法就是按照`DataFrame`的行或者列来进行排序，参数列表里面有`'by', 'axis', 'ascending', 'inplace', 'kind', 'na_position'`这几个参数，现在我们就来看一看每个参数是什么作用：

```python
>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame({
    'col1' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2' : [2, 1, 9, 8, 7, 4],
    'col3' : [0, 1, 9, 4, 2, 3]
})
>>> print(df)
  col1  col2  col3
0    A     2     0
1    A     1     1
2    B     9     9
3  NaN     8     4
4    D     7     2
5    C     4     3
```

这里定义了一个6行3列的`DataFrame`，其中有一个空值。

`axis`这个参数的默认值为`0`，匹配的是`index`，跨行进行排序，当`axis=1`时，匹配的是`columns`，跨列进行排序

`by`这个参数要求传入一个字符或者是一个字符列表，用来指定按照`axis`的中的哪个元素来进行排序

```python
>>> print(df.sort_values(by='col1'))
  col1  col2  col3
0    A     2     0
1    A     1     1
2    B     9     9
5    C     4     3
4    D     7     2
3  NaN     8     4
>>> print(df.sort_values(by=['col1', 'col2']))
  col1  col2  col3
1    A     1     1
0    A     2     0
2    B     9     9
5    C     4     3
4    D     7     2
3  NaN     8     4
```

`ascending`这个参数的默认值是`True`，按照升序排序，当传入`False`时，按照降序进行排列

```python
>>> print(df.sort_values(by='col1', ascending=False))
  col1  col2  col3
4    D     7     2
5    C     4     3
2    B     9     9
0    A     2     0
1    A     1     1
3  NaN     8     4
>>> print(df.sort_values(by='col1', ascending=True))
  col1  col2  col3
0    A     2     0
1    A     1     1
2    B     9     9
5    C     4     3
4    D     7     2
3  NaN     8     4
```

`kind`这个参数表示按照什么样算法来进行排序，默认值是`quicksort`（快速排序），也可以传入`mergesort`（归并排序）或者是`heapsort`（堆排序），至于具体每种算法是如何实现的，我们这里按下不表，同样的，对于`inplace`这个参数我们也不做讨论对于涉及到的`In-place algorithm`（原地算法）感兴趣的可以看看[这里](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/In-place_algorithm)
最后一个参数`na_position`是针对`DataFrame`中的空缺值的，默认值是`last`表示将空缺值放在排序的最后，也可以传入`first`放在最前：

```python
>>> print(df.sort_values(by='col1', ascending=False, na_position='first'))
  col1  col2  col3
3  NaN     8     4
4    D     7     2
5    C     4     3
2    B     9     9
0    A     2     0
1    A     1     1
>>> print(df.sort_values(by='col1', ascending=False, na_position='last'))
  col1  col2  col3
4    D     7     2
5    C     4     3
2    B     9     9
0    A     2     0
1    A     1     1
3  NaN     8     4
```

今天`pandas.DataFrame.sort_values`这个方法我们就讲到这里啦！其实`pandas.Series`也有`sort_values`方法，但是和`Dataframe.sort_values`的用法很接近，我就不赘述啦！有兴趣的同学可以去看看[文档](https://link.zhihu.com/?target=http%3A//pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.sort_values.html)，今天文章里涉及的代码都可以在我的[Github](https://link.zhihu.com/?target=https%3A//github.com/olivercqc/Python-Data-Analytics/blob/master/Pandas/pandas.DataFrame.sort_values.ipynb)里找到，文章和代码中如果出现了什么错误还烦请各位不吝赐教，批评指正！

