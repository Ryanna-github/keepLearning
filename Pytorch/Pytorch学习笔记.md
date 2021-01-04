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

## 

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
  - `torch.rom_numpy(numpy_array)`：共享内存

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



