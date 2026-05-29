# 训练主程序main()的完整逻辑

## 一些准备工作

获取超参数信息（通过自定义的parse_args和device函数）

设置模型参数（transformer_lm）

快速统计模型参数数量的方法：

total_params = sum(p.numel() for p in model.parameters())

## Training Loop

1. 确定单轮“超参数”：lr
2. data_loading加载数据
3. 数据搬至gpu
4. 开启前向传播过程：（1）初始化优化器（梯度归零）；（2）数据输入模型；（3）计算损失函数值
5. 开启反向传播过程：（1）计算损失函数关于个参数的梯度值（loss.backward）:(3)防止梯度过大进行梯度截断；（4）优化器完成迭代（注意下一轮还要先初始化优化器，这也是要先Optimizer.zero_grad()）

训练过程的验证（Validation）：采用指数型困惑度

定期保存checkpoint