from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from datasets import load_dataset

UNK, PAD, SOS, EOS = '<unk>', '<pad>', '<sos>', '<eos>'
SPECIALS = [UNK, PAD, SOS, EOS]


class Vocab:
    def __init__(self, counter:Counter, min_freq: int=2) -> None:
        self.itos: List[str] = list(SPECIALS)
        for tok, cnt in counter.most_common():
            if cnt>=min_freq and tok not in SPECIALS:
                self.itos.append(tok)
        self.stoi = {tok: i for i , tok in enumerate(self.itos)}
        self.unk_idx = self.stoi[UNK]
        self.pad_idx = self.stoi[PAD]
        self.sos_idx = self.stoi[SOS]
        self.eos_idx = self.stoi[EOS]
    
    def __len__(self) -> int:
        return len(self.itos)
    
    def __call__(self,tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]
    
    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

class Multi30kDataset:

    _spacy_de = None
    _spacy_en = None

    def __init__(self, split='train', src_vocab: Vocab = None, tgt_vocab: Vocab =None, min_freq: int=2, max_len: int=100) -> None:
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.max_len = max_len
        
        if Multi30kDataset._spacy_de is None:
            Multi30kDataset._spacy_de = spacy.load('de_core_news_sm')
            Multi30kDataset._spacy_en = spacy.load('en_core_web_sm')
         
        # Loading Raw Data
        ds = load_dataset('bentrevett/multi30k', split=split)
        self.pairs: List[Tuple[str,str]] = [(ex['de'], ex['en']) for ex in ds]

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self._build_vocab(min_freq)
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        
        self.data = self._process_data()

        # Tokenization

    def _tok_de(self, text: str) -> List[str]:
        return[t.text.lower() for t in self._spacy_de.tokenizer(text)]
        
    def _tok_en(self, text: str) -> List[str]:
        return[t.text.lower() for t in self._spacy_en.tokenizer(text)]
    
    # Vocabulary Construction
        
    def build_vocab(self):
        return self._build_vocab()

    def _build_vocab(self, min_freq: int=2) -> Tuple[Vocab,Vocab]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter, tgt_counter = Counter(), Counter()
        for de, en in self.pairs:
            src_counter.update(self._tok_de(de))
            tgt_counter.update(self._tok_en(en))
        return Vocab(src_counter, min_freq), Vocab(tgt_counter, min_freq)
    
    # Interger encoding

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        return self._process_data()
    
    def _process_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for de, en in self.pairs:
            src = self._tok_de(de)[: self.max_len]
            tgt = self._tok_en(en)[: self.max_len]
            src_ids = [self.src_vocab.sos_idx] + self.src_vocab(src) + [self.src_vocab.eos_idx]
            tgt_ids = [self.tgt_vocab.sos_idx] + self.tgt_vocab(tgt) + [self.tgt_vocab.eos_idx]
            out.append((torch.tensor(src_ids, dtype=torch.long),torch.tensor(tgt_ids, dtype=torch.long)))
        return out
        

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.data[idx]
    
    # Padding a batch to the longest sequence

def collate_fn(batch, pad_idx: int=1):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first =True, padding_value =pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first =True, padding_value =pad_idx)
    return src_batch, tgt_batch