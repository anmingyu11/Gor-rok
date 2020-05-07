## 在multiIndex中选定指定索引的行

我们在用pandas类似groupby来使用多重index时，有时想要对多个level中的某个index对应的行进行操作，就需要在dataframe中找到该index对应的行，在单层index中我们可以方便的使用df.loc[index]来选择，在多重Index中我们可以利用的类似的思路，然而其中也有一些小坑，记录如下。

### 1. index为有序的
#### 1.1 创建测试数据

首先创建一个dataframe数据

```python
df = pd.DataFrame({'class':['A','A','A','B','B','B','C','C'],
                   'id':['a','b','c','a','b','c','a','b'],
                   'value':[1,2,3,4,5,6,7,8]})
```

df中内容如下图：

![df](/Users/helloword/Anmingyu/Gor-rok/PY/MultiIndex/1.png)

#### 1.2 设置multiIndex

通过**set_index**设为多重索引

```python
df = df.set_index(['class','id'])
```

设置索引后效果：

![df_multiindex](/Users/helloword/Anmingyu/Gor-rok/PY/MultiIndex/2.png)



#### 1.3 切片筛选index

这里同样使用loc定位

```python
df.loc[('A',slice(None)),:]
```

各参数的解释如下：

- `loc[(a,b),c]`中第一个参数元组为索引内容，`a`为`level0`索引对应的内容，`b`为`level1`索引对应的内容因为`df`是一个`dataframe`，所以要用`c`来指定列
- 这里`A`，指选择class中的A类
- `slice(None)`, 是Python中的切片操作，这里用来选择任意的`id`，要注意！不能使用`:`来指定任意index
- `:`,用来指定dataframe任意的列

执行后的结果如下：

同样，如果想只保留id中的`a`，则可以使用：

```python
df.loc[(slice(None),'a'),:]
```

### 2 index无序

前面的例子对应的`index`列为数字或字母，是有序的，接下来我们看看`index`列为中文的情况。

#### 2.1 创建无序测试数据

```python
df2 = pd.DataFrame({'课程':['语文','语文','数学','数学'],'得分':['最高','最低','最高','最低'],'分值':[90,50,100,60]})
df2 = df2.set_index(['课程','得分'])
```

#### 2.2 尝试切片选择index

```python
df2.loc[('语文',slice(None)),:]
```

我们进行同样的操作，这时会发现提示出错：

```
UnsortedIndexError: 'MultiIndex Slicing requires the index to be fully lexsorted tuple len (2), lexsort depth (0)'
```

这是因为此时的`index`无法进行排序，在pandas文档中提到：`Furthermore if you try to index something that is not fully lexsorted, this can raise:`

我们可以通过 `df2.index.is_lexsorted()`来检查index是否有序，

```python
 In[1]: df2.index.is_lexsorted()
out[1]: False
```


接下来，我们尝试对Index进行排序。（排序时要在level里指定index名）

#### 2.3 对index排序后切片选择index

```python
df2 = df2.sort_index(level='课程')
df2.loc[('语文',slice(None)),:]
```

![结果](/Users/helloword/Anmingyu/Gor-rok/PY/MultiIndex/3.png)

得到了我们想要的结果。

