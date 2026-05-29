# generate_text

 输入: "Once upon a time"
    ↓
[编码] tokenizer.encode()
    ↓
[初始] input_ids = [1, 42, 1337, 99]  # token IDs
    ↓
┌─────────────────────────────────┐
│  自回归生成循环（max_tokens次）    │
├─────────────────────────────────┤
│ 1. 模型前向传播 → logits          │
│ 2. 取最后一个位置的logits         │
│ 3. Temperature缩放               │
│ 4. Softmax → 概率分布             │
│ 5. Top-p采样（可选）              │
│ 6. 从概率分布中采样下一个token     │
│ 7. 添加到序列                     │
│ 8. 检查是否遇到EOS                │
│ 9. 更新input_ids用于下次预测       │
└─────────────────────────────────┘
    ↓
[解码] tokenizer.decode()
    ↓
输出: "Once upon a time there lived a beautiful princess..."

**8. 检查是否遇到EOS**：用到.strip()为了防止字符串多余的空白字符：空格、换行符、制表符等。


# load_model_and_tokenizer（模型入口）

加载训练好的模型和分词器，是模型部署和推理的入口函数


为什么用 try-except？：

1. 提供容错性：如果导入路径不对，不会直接崩溃
2. 支持渐进开发：可以先测试模型加载，后实现分词器

## 记载组件汇总
组件	作用	关键点
分词器	文本 ↔ token ID	提供容错的导入逻辑
检查点	保存的模型权重	支持不同的存储格式
配置	模型架构参数	优先使用检查点中的配置
设备	CPU/GPU 加载	使用 map_location
权重加载	恢复模型参数	处理前缀和严格模式

## 补充知识（about 代码工程化）

python内置函数，动态导入模块
```
__import__(path, fromlist=['Tokenizer'])
```

动态导入的示例：
```
def dynamic_import_example():
    """演示动态导入的各种用法"""
    
    # 方式1：使用 __import__
    module = __import__('math')
    print(module.sqrt(9))  # 3.0
    
    # 方式2：导入子模块
    module = __import__('os.path', fromlist=['join'])
    print(module.join('/home', 'user'))  # /home/user
    
    # 方式3：导入特定属性
    module = __import__('math', fromlist=['pi'])
    pi = getattr(module, 'pi')
    print(pi)  # 3.14159...
    
    # 方式4：处理可能不存在的模块
    module_names = ['numpy', 'pandas', 'nonexistent_module']
    
    for name in module_names:
        try:
            module = __import__(name)
            print(f"✓ {name} available")
            
            # 检查特定函数是否存在
            if hasattr(module, 'array'):
                print(f"  - Has array function")
        except ImportError:
            print(f"✗ {name} not available")
```

动态导入详解
动态导入是指在程序运行时根据条件或字符串来导入模块，而不是在代码开头用 import 语句静态导入。

# main()

执行命令的总程序：

```
parser.add_argument('--device', type=str, default='auto', 
                   help='Device: auto, cpu, cuda, mps')
```

default参数为运行命令行时没有输入参数时的默认参数；
help为帮助信息



