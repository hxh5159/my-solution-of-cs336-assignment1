import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None 
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_2_id_vocab = {v:k for k,v in vocab.items()}
        self.merges_ranking = {merge:idx for idx, merge in enumerate(merges)}
        #finanlly we need to handle special tokens
        if special_tokens:
            self.special_tokens_2_bytes = [token.encode('UTF-8') for token in special_tokens]
        else:
            self.special_tokens_2_bytes = []
        # 通过循环结构将特殊的词节补充在self.byte_2_id_vocab中
        for special_token in self.special_tokens:
            if special_token not in self.byte_2_id_vocab:
                n=len(self.vocab)
                self.vocab[n]=special_token
                self.byte_2_id_vocab[special_token]=n
    # 通过序列化文件加载Tokenizer的类方法
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method that loads a tokenizer from serialized files.

        Args:
            vocab_filepath: Path to a pickle file containing a dict[int, bytes] (or bytes-like).
            merges_filepath: Path to a pickle file containing a list[tuple[bytes, bytes]] (or str-like).
            special_tokens: Optional list[str] to be registered/appended to the vocabulary.

        Returns:
            An initialized Tokenizer instance.
        """
        import pickle

        # Load and normalize vocab: keys -> int, values -> bytes
        with open(vocab_filepath, "rb") as vf:
            raw_vocab = pickle.load(vf)

        norm_vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            kid = int(k)
            if isinstance(v, str):
                v = v.encode("utf-8")
            norm_vocab[kid] = v

        # Load and normalize merges: ensure tuples of bytes
        with open(merges_filepath, "rb") as mf:
            raw_merges = pickle.load(mf)

        norm_merges: list[tuple[bytes, bytes]] = []
        for a, b in raw_merges:
            if isinstance(a, str):
                a = a.encode("utf-8")
            if isinstance(b, str):
                b = b.encode("utf-8")
            norm_merges.append((a, b))

        return cls(norm_vocab, norm_merges, special_tokens)
    

    
    def encode(self, text: str) -> list[int]:
        res_token_ids = []
        pretokenizeiton = self.pre_tokenization(text, self.special_tokens)  # 对文本进行预分词
        # 如果是特殊token，则直接映射到id，如果不是直接通过encode_text进行编码
        for part in pretokenizeiton:
            if self.special_tokens and part in self.special_tokens:
                special_id = self.byte_2_id_vocab[part.encode('UTF-8')]
                res_token_ids.append(special_id)
            else:
                res_token_ids.extend(self.encode_text(part))
        return res_token_ids
    
    # 不一次性添加，而是迭代的添加token_id，保证内存占用稳定
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    def decode(self, ids: list[int]) -> str:
        #transform to byte_list
        byte_list = b''.join(self.vocab[id_] for id_ in ids)
        return byte_list.decode('UTF-8', errors='replace')
    
    def encode_text(self, pre_token: str):
        '''
        encode a single pre-token (normal text, not special tokens) to token ids
        '''
        def word_2_byte(word: str) -> tuple[bytes, ...]:
            word_decoded = list(word.encode('UTF-8'))
            word_byte = [bytes([b]) for b in word_decoded]
            return tuple(word_byte)
        
        word_byte = word_2_byte(pre_token)
        word_byte_after_merge = self.apply_merge(word_byte)  # 应用合并规则，从单个字节开始合并，注意合并的顺序
        token_ids = []
        for merged_bytes in word_byte_after_merge:
            id_ = self.byte_2_id_vocab[merged_bytes]
            token_ids.append(id_)
        return token_ids
