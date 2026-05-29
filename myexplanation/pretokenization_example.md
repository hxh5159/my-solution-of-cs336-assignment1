这部分的作用在与对大规模的语料库进行分块以便多进程处理，注意分块的一个问题是要保证特殊token(标记文本结束的token)要出现在每个块的最后（要根据语料库的实际情况对boundary进行偏移）

**注意data_loading是在cpu中完成的**

主要流程：
1. 计算初始边界
2. 调整边界到特殊token

偏移的大致逻辑：

# 从位置 33 开始读取
initial_position = 33
file.seek(33)  # 在 "dog<|endoftext|>" 中间

# 读取 4KB
mini_chunk = file.read(4096)  # 包含 "dog<|endoftext|>Hello..."

# 查找 b'<|endoftext|>'
found_at = mini_chunk.find(b'<|endoftext|>')  # 找到，偏移 3

# 新边界 = 33 + 3 = 36
chunk_boundaries[1] = 36  # 在 <|endoftext|> 之后

偏移boundary后的结果：
文件内容（横轴为字节位置）：
0        36       66      100
|--------|--------|--------|
| 块1    | 块2    | 块3    |

详细内容：
[0-36]   The quick brown fox<|endoftext|>
[36-66]  jumps over the lazy dog<|endoftext|>
[66-100] Hello world<|endoftext|>Goodbye<|endoftext|>

注意：块边界总是在 <|endoftext|> 之后
这样可以确保每个块以完整文档结束
3. 创建多个进程并进行并行处理
4. 合并结果

形如：

chunks = [(file, 0, 100), (file, 100, 200), (file, 200, 300)]
         ↓
    with mp.Pool(3) as pool:  # 创建3个进程
         ↓
    pool.starmap(func, chunks)  # 并行执行
         ↓
    results = [result1, result2, result3]  # 收集结果
         ↓
    合并所有结果