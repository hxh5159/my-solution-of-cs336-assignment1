
# class Tokenizer

## __init__初始化

```
for special_token in self.special_tokens_2_bytes:
    if special_token not in self.byte_2_id_vocab:
        n = len(self.vocab)
        self.vocab[n] = special_token
        self.byte_2_id_vocab[special_token] = n
```

运行示例
```
# 假设初始 vocab 大小是 1000
len(self.vocab) = 1000

# 添加第一个特殊 token
n = 1000
self.vocab[1000] = b'<|endoftext|>'
self.byte_2_id_vocab[b'<|endoftext|>'] = 1000

# 添加第二个特殊 token
n = 1001
self.vocab[1001] = b'<|unk|>'
self.byte_2_id_vocab[b'<|unk|>'] = 1001
```

Tokenizer的总体逻辑：

创建和使用Tokenizer
```
# 示例数据
vocab = {
    0: b'a',
    1: b'b',
    2: b'c',
    3: b'ab',
}

merges = [
    (b'a', b'b'),  # 合并 a 和 b → ab
]

special_tokens = ['<|endoftext|>', '<|unk|>']

# 创建 tokenizer
tokenizer = Tokenizer(vocab, merges, special_tokens)

print("Vocabulary (ID → bytes):")
for id, bytes_val in tokenizer.vocab.items():
    print(f"  {id}: {bytes_val}")

print("\nReverse mapping (bytes → ID):")
for bytes_val, id in tokenizer.byte_2_id_vocab.items():
    print(f"  {bytes_val}: {id}")

print("\nMerge rankings:")
for merge, rank in tokenizer.merges_ranking.items():
    print(f"  {merge}: rank {rank}")

print("\nSpecial tokens (bytes):")
for token in tokenizer.special_tokens_2_bytes:
    print(f"  {token}")
```

输出
```
Vocabulary (ID → bytes):
  0: b'a'
  1: b'b'
  2: b'c'
  3: b'ab'
  4: b'<|endoftext|>'
  5: b'<|unk|>'

Reverse mapping (bytes → ID):
  b'a': 0  # 这里的b表明是字节而不是一般的字符串
  b'b': 1
  b'c': 2
  b'ab': 3
  b'<|endoftext|>': 4
  b'<|unk|>': 5

Merge rankings:
  (b'a', b'b'): rank 0

Special tokens (bytes):
  b'<|endoftext|>'
  b'<|unk|>'
```

## tokenizer的逻辑
```
encode_text() → apply_merge() → 返回 token IDs
     ↓
编码单个单词
     ↓
使用 BPE 合并规则
```

## encode_text方法

 工作流：

输入: "lower"
    ↓
word_2_byte() 将单词转换为字节元组
    ↓
(l'b', b'o', b'w', b'e', b'r')
    ↓
apply_merge() 应用 BPE 合并规则
    ↓
(b'low', b'er')
    ↓
byte_2_id_vocab 查找 ID
    ↓
[256, 257]
    ↓
返回 Token IDs

##  bpe的合并逻辑

# 以lower为例

# 初始
word = [b'l', b'o', b'w', b'e', b'r']
word_pairs = {(b'l',b'o'), (b'o',b'w'), (b'w',b'e'), (b'e',b'r')}

# 第1轮：合并 (b'l',b'o')
bigram = (b'l',b'o')
→ word = [b'lo', b'w', b'e', b'r']
→ word_pairs = {(b'lo',b'w'), (b'w',b'e'), (b'e',b'r')}

# 第2轮：合并 (b'lo',b'w')
bigram = (b'lo',b'w')
→ word = [b'low', b'e', b'r']
→ word_pairs = {(b'low',b'e'), (b'e',b'r')}

# 第3轮：合并 (b'e',b'r')
bigram = (b'e',b'r')
→ word = [b'low', b'er']
→ word_pairs = {(b'low',b'er')}

# 第4轮：合并 (b'low',b'er')
bigram = (b'low',b'er')
→ word = [b'lower']
→ word_pairs = set() (只有一个元素)

# 循环结束
return [b'lower']

# pre_tokenization的逻辑

输入: "Hello<|endoftext|>world! [SEP] 42"
special_token: ['<|endoftext|>', '[SEP]']
    │
    ├─→ 排序特殊 token (长优先)
    │   ['<|endoftext|>', '[SEP]']
    │
    ├─→ 构建正则: "(<\|endoftext\|>|\[SEP\])"
    │
    ├─→ re.split() 分割
    │   ['Hello', '<|endoftext|>', 'world! ', '[SEP]', ' 42']
    │
    └─→ 处理每个 part
        ├─ 'Hello' → 正则分词 → ['Hello']
        ├─ '<|endoftext|>' → 特殊 token → ['<|endoftext|>']
        ├─ 'world! ' → 正则分词 → ['world', '!', ' ']
        ├─ '[SEP]' → 特殊 token → ['[SEP]']
        └─ ' 42' → 正则分词 → [' 42']
    
最终: ['Hello', '<|endoftext|>', 'world', '!', ' ', '[SEP]', ' 42']


主要作用是保护特殊的token和保留空格、标点等语义边界