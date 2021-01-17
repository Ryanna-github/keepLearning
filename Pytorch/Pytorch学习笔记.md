# Tensor 操作

## 赋值

```python
import torch
# 未初始化
x = torch.empty(row_number, col_number)
# 随机初始化
x = torch.rand(row_number, col_number)
# 初始化为0 & 指定类别
x = torch.zeros(row_number, col_number, dtype = torch.long)
# 转化已有数据集
x = torch.tensor([...])
# 相同 dtype, device 属性的赋值
x = x.new_ones(row_number, col_number, dtype = torch.float64)
# 指定新的数据类型
x = torch.randn_like(x, dtype = torch.float)
```

其他：对角线，均匀切分，正态分布，随机排列等



## 形状相关

- 指定维数为 `-1`表示由其他维度推出该维度
- `reshape()` 并不能保证返回的是拷贝

```python
# 获取
x.size()
x.shape
# 改变
## 方法1：view()，共享内存
y = x.view(15)
y = x.view(-1, 5)
## 方法2：clone()，真正副本
y = x.clone().view(15)
# 获取 tensor 中仅有的单个数字
x.item()
```



## 基本操作

```{python}
# 加法
x = torch.rand(5, 3)
y = torch.rand(5, 3)
## 方法1
x + y
## 方法2（可以指定输出结果）
torch.add(x, y) 
result = torch.empty(5, 3)
torch.add(x, y, out = result)
## 方法3（替换，后缀为_）
y.add_(x)
```



## 内存

- 索引得到的结果共享内存，索引得到结果赋给新变量，改变新变量原 tensor 也会被修改

- 可以用 `id(x)` 检查边浪的内存地址，在记不清某操作是否共享内存时可以这样进行验证

```{python}
# 前后两个 y id 不相同
y = y + x
# 前后两个 y id 相同
y[:] = y + x
y = torch.add(x, y, out = y)
y += x
y.add_(x)
```



## 转化

### 数据类型转化

- Tensor 转为 NumPy
  - `t.numpy()`：共享内存
  - `torch.tensor(numpy_array)`：分配新内存

- Numpy 转为 Tensor
  - `torch.from_numpy(numpy_array)`：共享内存

### 所在设备转化

```{python}
# 检查 cuda 是否可用
torch.cuda.is_available()
# 之间指定设备创建
device = torch.device("cuda")
y = torch.ones_like(x, device = device)
# 转移
x = x.to(device)
# 转移时更改数据类型
x = x.to("cpu", torch.double)
```



## 其他

- 设置默认数据类型

```{python}
torch.set_default_tensor_type(torch.FloatTensor)
```




# 梯度

- 通过 Tensor 的属性调整确定是否追踪梯度
  - 追踪：设置 Tensor 属性 `.requires_grad = True`
  - 分离：对于单个 Tensor 可以用`.detach()`，或直接对某一代码块 `with torch.no_grad()`
- 直接创建的 Tensor 没有 `grad_fn`，通过运算操作创建的 Tensor 有该属性
  - 直接创建的称为叶子节点，检查方式 `x.is_leaf`
  - 注意区分 `requires_grad` 和 `grad_fn` 属性的区别
- 只有 `requires_grad = True`时候才可以调用 `.backward()`

## 追踪

```{python}
# 创建时就指明要计算梯度
x = torch.ones(row_number, col_number, requires_grad = True)
# 改变属性
x.requires_grad_(True)
```

## 计算

- 求导结果是一个和自变量同形的张量
  - 必要的时候把张量通过所有元素加权求和的方式转化为标量
  - 权，即为传入的参数，与输出变量同形，先计算内积转化为标量，在得到和自变量同形的梯度向量
- 对于标量可以直接进行反向传播

```{python}
# x, y, z 均 erquirs_grad = True
# y, z 的 grad_fn 不是 None
x = torch.ones(2, 2, requires_grad = True)
y = x + 2
z = y * y * 3
out = z.mean()
# out 是标量，直接调用 backward 求导
out.backward()
x.grad
```

- grad 在反向传播过程中是累加的，注意在反向传播之前都需要把梯度清零
- 如果有中断追踪的部分，反向传播过程中不会将这一小部分的梯度计算在内，其他部分照常计算

```{python}
x.grad.data.zero_()
out.backward()
x.grad
```

## 修改 Tensor（不反向传播）

- 对于图中记录的修改，`requires_grad = True` Tensor 改变会引起相关的反向传播梯度向量的改变
- 如果想强行改变数值但是不影响反向传播的梯度

```{python}
x = torch.ones(1, requires_grad = True)
y = 2 * x
# 改变 tensor 值
x.data *= 100
y.backward()
# 梯度不会改变
x.grad
```



## 实用函数

### 聚合

```{python}
# 方法1
torch.gather(data_tensor, dim, index_tensor)
# 方法2
data_tensor.gather(dim, index_tensor)
```

- `index_tensor 需要是 `LongTensor` 类
- `data_tensor` 和 `index_tensor` 型完全相同
- 理解：`data_tensor` 在指定的维度取 `index_tensor` 对应位置的下标的元素

---

# 激活函数

> 计算方式：对于向量应用某种激活函数后，先对结果求和再进行反向传播

- Relu
- Sigmoid
- Tanh



---



# 其他注意

- 神经网络中只有设计到计算才可以算作一层，比如线性回归是单层神经网络，因为输入层没有涉及任何计算

---



# 建模工具

## 数据集

### 使用自定义数据集

开始前需要准备好 data 和 labels，两个 Tensor 的行数应当相同，设定好 batch_size 后可以直接调用 `torch.utils.data` 模块以简化操作

```{python}
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)
```

- 其中`dataset.tensors[0]` = features，`dataset.tensors[1]` = labels

- `data_iter` 是一个生成器，已处理最后一个 batch 取不满 batch_size 的情况

```{python}
# 遍历所有 batch
for X, y in data_iter:
    print(X, y)
# 取一个 batch（测试时常用）
X, y = iter(test_iter).next()
```

- `X` 为一个 batch 的 data，`y` 即为其对应 label
- `DataLoader` 允许多进程加速数据读取，对应参数设置为 `num_workers`
  - 注意，如果不需要额外的进程来加速数据的读取则取 0

### 下载数据集

```{python}
# 直接下载
mnist_train = torchvision.datasets.FashionMNIST(root = '/data', train = True, download = True, transform = transforms.ToTensor())
# 加载已经下载到目标目录的数据集
mnist_train = torchvision.datasets.FashionMNIST(root = './data/', train = True, transform = transforms.ToTensor())
```



## 模型

利用模块 `torch.nn`，可以很方便自定义网络或者对已有网络结构进行改动，以线性回归为例，具体使用方法如下

```{python}
import torch.nn as nn

## Method 1
net = nn.Sequential(nn.Linear(num_inputs, 1))

## Method 2 (with customized layer name)
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

## Method 3: (with customized layer name)
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
]))
```

- 三种方法都具有等价的拓展性
- 以上可以作为继承 `nn.Module` 类的新类网络网络结构定义
- 注意：`torch.nn` 仅支持以 batch 为形式的输入，如果需要传入单个样本（比如得到单个样例的输出结果）需要 `input.unsqueeze(0)` 增加维度



## 参数

以上方式定义的网络，可以调用 `net.parameters()` 查看所有可学习的参数

```{python}
# 查看:net.parameters()
for param in net.parameters():
    print(param)

# 初始化
from torch.nn import init
init.normal_(net[0].weight, mean = 0, std = 0.01)
init.constant_(net[0].bias, val = 0)
net[0].bias.data.fill_(0) # 与上一行等价
```

- `net.parameters()` 直接取的是参数内存而不是拷贝，可以直接传入优化器进行优化
- 注意：pytorch 中模块已经将参数初始化，不需要再次手动初始化



## 损失函数

### 使用内置损失函数

依旧使用 `torch.nn` 模块，例如

```{python}
loss = nn.MSELoss()
```

- 平方损失函数：`nn.MSELoss()`
  - 回归任务中常用
  - 计算公式：$$loss = \sum_{i = 1}^{n}({\hat{y}}^{(i)}-y^{(i)})/2$$
- 交叉熵损失函数：`nn.`
  - 多分类任务中常用，只关心预测的正确与否而非得到判断结果的把握有多大
  - 计算公式：$loss = -\sum_{i = 1}^{n}y_j^{(i)}\log{\hat{y_j}^{(i)}} = -\sum_{i = 1}^{n}\log_{y_j^{(i)}}{\hat{y_j}^{(i)}}$

### 自定义损失函数

注意：只要向量 `requires_grad = True	` 那么就可以调用 `backward()` 函数，可以实现根据损失函数得到梯度，进而再调用优化器对模型参数进行调整

 ## 优化算法

使用模块：`torch.optim`

### 固定学习率

```{python}
import torch.optim as optim 

# 常用方法
optimizer = optim.SGD(net.parameters(), lr = 0.03)
print(optimizer) # print 得到的结果都可以在参数中修改

# 精细调整（不同结构学习率不同）
optimizer = optim.SGD([
    {'params': net.subnet1.parameters()},
    {'params': net.subnet2.parameters(), 'lr': 0.01}
], lr = 0.03)
```

### 调整学习率

- 方法一：`optimizer.param_groups` 可以查看每个可学习的参数当前对应的学习率，可以对其修改以实现学习率调整

- 方法二：构建新的优化器
  - 优化器的构建开销很小，旧的不好不妨直接换新
  - 但是这样会损失动量等状态信息，损失函数收敛会出现震荡

```{python}
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1
```

### 其他设置

- 权重衰减

```{python}
optimizer = torch.optim.SGD(params = net.parameters(), lr = 0.05, weight_decay = True)
```

- 可以对不同参数设置不同的优化器对参数进行不同的调整，比如可以设定权重项进行权重衰减但是偏置项不进行衰减





## 模型训练

```{python}
optimizer.zero_grad()
loss.backward()
optimizer.step()
```



## 参数更新

- 一般操作：$w_{1} \leftarrow\left(1-\frac{\eta \lambda}{|\mathcal{B}|}\right) w_{1}$

- 权重衰减：$L_2$ 范数正则化衰减权重

  $w_{1} \leftarrow\left(1-\frac{\eta \lambda}{|\mathcal{B}|}\right) w_{1}-\frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} x_{1}^{(i)}\left(x_{1}^{(i)} w_{1}+x_{2}^{(i)} w_{2}+b-y^{(i)}\right)$
  - 其损失函数在原有基础上添加 $\frac{\lambda}{2n}||\omega||^2$，即对复杂模型进行惩罚
  - $\lambda>0$，值越大权重学习到的权重元素越接近零
  - 可以通过设置 `optimizer()` 中的 `weight_decay = True` 