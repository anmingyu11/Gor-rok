# 读写JSON数据

## 问题

你想读写JSON(JavaScript Object Notation)编码格式的数据。

## 解决方案

`json` 模块提供了一种很简单的方式来编码和解码JSON数据。 其中两个主要的函数是 `json.dumps()` 和 `json.loads()` ， 要比其他序列化函数库如`pickle`的接口少得多。 下面演示如何将一个Python数据结构转换为JSON：

```python
import json

data = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23
}

json_str = json.dumps(data)
```

下面演示如何将一个JSON编码的字符串转换回一个Python数据结构：

```python
data = json.loads(json_str)
```

如果你要处理的是文件而不是字符串，你可以使用 `json.dump()` 和 `json.load()` 来编码和解码JSON数据。例如：

```python
# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)
```



-----------------



```python
item_4categories_map = {
    'cate1_code' : 'category',
    'cate2_code' : 'category',
    'cate3_code' : 'category',
    'cate4_code' : 'category',
    'cate_en_name' : 'category',
    'cate_ch_name' : 'category'
}
# Writing JSON data
with open('./data/item_4categories_datamap.json', 'w') as f:
    json.dump(item_4categories_map, f)

# Reading data back
with open('./data/item_4categories_datamap.json', 'r') as f:
    item_4categories_map = json.load(f)
```

