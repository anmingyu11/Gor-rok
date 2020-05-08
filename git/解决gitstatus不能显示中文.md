解决git status不能显示中文

- 现象
status查看有改动但未提交的文件时总只显示数字串，显示不出中文文件名，非常不方便。如下图：


- 原因
在默认设置下，中文文件名在工作区状态输出，中文名不能正确显示，而是显示为八进制的字符编码。

- 解决办法
将git 配置文件`core.quotepath`项设置为false。
quotepath表示引用路径
加上--global表示全局配置

git bash 终端输入命令：

```
git config --global core.quotepath false
```
