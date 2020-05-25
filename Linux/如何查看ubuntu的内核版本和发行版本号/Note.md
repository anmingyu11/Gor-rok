# 如何查看ubuntu的内核版本和发行版本号？

有时候，我们在升级内核版本或者是从一个版本升级到新的版本之后，想要查看一下自己的ubuntu是否升级成功。可是有没有一种比较快捷的方法比如说在终端里面查看呢？答案是肯定的。为了查看 Ubuntu 的版本号，可以采用以下两种方法之一。

### 方法一

在终端中执行下列指令：

```
cat /etc/issue
```

可以查看当前正在运行的 Ubuntu 的版本号。其输出结果类似下面的内容：

```
Ubuntu 8.04 /n /l
```

### 方法二

使用 lsb_release 命令也可以查看 Ubuntu 的版本号，与方法一相比，内容更为详细。执行指令如下：

```
sudo lsb_release -a
```

将输出结果：

Distributor ID:    Ubuntu
Description:    Ubuntu 8.04
Release:    8.04
Codename:    hardy

# 查看内核版本号

查看内核版本号的方法是：
打印一个终端，输入命令

```
uname -r
```