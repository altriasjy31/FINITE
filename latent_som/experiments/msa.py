import os
import random
import sys
import pathlib
prj_path = str(pathlib.Path(__file__).parent.parent)
if prj_path not in sys.path:
   sys.path.append(prj_path)

import helper_functions.constants as C

import pickle
import torch
import torch as th
import torch.nn as nn
import torch.utils.data as td
import torch.utils.data.dataset as D
import torch.nn.functional as F
import typing as T
import string
import re
import itertools as it
import more_itertools as mit
import numpy as np
import Bio.SeqIO as sio
import Bio.SeqRecord as R

DATA = T.Dict[str, T.Dict[str, T.Dict[str, T.List[T.Union[T.List[int], str]]]]]
NOLOWER = str.maketrans('','', string.ascii_lowercase)
SPILTER = re.compile(">[^>\n]+\n")
MAPPING = np.zeros(256,np.uint8) # unsigned int8, 0-255
alphabet = np.frombuffer(C.trR_ALPHABET.encode(), np.uint8)
index_alphabet = np.arange(len(C.trR_ALPHABET))
MAPPING[alphabet] = index_alphabet

class MSADataset(D.Dataset):
  namekey = "proteins"
  labelkey = "prop_annotations"
  num_classes=40000
  classes=th.zeros(num_classes, dtype=th.int)
  msa_format = "a3m"
  def __init__(self, 
               dataset: str, msa_dir: str,
               mode: str, task: str, 
               num_classes: int, 
               msa_buffer_size: int,
               max_len: int,
               cutoff: float = 0.8,
               eps: float = 1e-9,
               need_proteins: bool = False,
               **kwargs):
    """
    """
    if (dataset.endswith(".pt")):
        data = torch.load(dataset)
    elif (dataset.endswith(".pkl")):
        with open(dataset, "rb") as h:
            data = pickle.load(h)
    else:
       raise NotImplementedError("not imp")
    
    self.num_classes = num_classes
    self.msa_buffer_size = msa_buffer_size
    self.max_len = max_len
    self.cutoff = cutoff
    self.eps = eps
    self.need_proteins = need_proteins

    self.msa_max_size = kwargs.get("msa_max_size", -1)

    self.targets = self._load_data(data, mode, task, msa_dir)
    self.len = len(self.targets["msa"])
    print(f"Loaded {mode}-set, length: {self.len}")
  
  def __len__(self):
     return self.len

  def reset_mode(self, mode):
    self.targets = self._load_data(data, mode, task, msa_dir)
    self.len = len(self.targets["msa"])
    print(f"Loaded {mode}-set, length: {self.len}")
  
  def _load_data(self,
                data: DATA,
                mode: str,
                task: str,
                msa_dir: str,
                msa_format: T.Optional[str] = None):
    """
    """

    if msa_format is None:
       msa_format = self.msa_format

    subdata = data[mode][task]
    subdata["msa"] = [p for x in subdata[self.namekey]
                     if os.path.exists(p := os.path.join(msa_dir, f"{x}.{msa_format}"))]
    
    return subdata

  def get(self, key, index):
      return self.targets[key][index]
    
  def __getitem__(self, index):
    proteins = self.get(self.namekey, index)
    msa_path = self.get("msa", index)
    assert isinstance(msa_path, str)

    a3m_lines = load_from(msa_path, self.msa_max_size)
    msa = th.from_numpy(encoder(a3m_lines)).long() # as long tensor
    # shuffle
    current_size = msa.size(0)
    idx = list(range(1,current_size))
    random.shuffle(idx)
    msa = msa[[0] + idx] # query sequence not change

    # select sequence for feature extraction
    msa_buffer_size = self.msa_buffer_size if self.msa_buffer_size != -1 else current_size

    # msa_buffer_size and max_len are used for computation cost control
    # padding for max_len
    current_len = msa.size(1)
    pad_len = self.max_len - current_len if current_len < self.max_len else 0
    pad_size = self.msa_buffer_size - current_size if current_size < msa_buffer_size else 0
    f1d = F.pad(msa, (0, pad_len, 0, pad_size))[:self.msa_buffer_size, :self.max_len]

    y = self.get(self.labelkey,index)
    assert isinstance(y, T.List)

    self.classes.fill_(0)
    self.classes[y] = 1

    if not self.need_proteins:
        return f1d, self.classes[:self.num_classes].clone()
    else:
        return (proteins, f1d), self.classes[:self.num_classes].clone()

def bufferedAlignReader(a3m_file: str, max_size: int = 10000):
  max_lines = max_size * 2
  with open(a3m_file, "r") as h:
    for i, line in enumerate(h, start=1):
      if i > max_lines:
        break
      yield line

def get_sequence(seqrecord: R.SeqRecord):
   return str(seqrecord.seq)

def load_from(a3m_file: str, max_size: int = 10000):
  """
  max_size: maximum sequence amount in each file
  """

  # with open(a3m_file, "r") as h:
    # return list(mit.take(2*max_size, h)) if max_size != -1 else list(h)
  if max_size != -1:
    return list(bufferedAlignReader(a3m_file, max_size))
  else:
    with open(a3m_file, "r") as h:
      return h.readlines()

def sequence2array(seq: str):
  return np.frombuffer(seq.encode(), dtype=np.uint8)

def build_array(seqs: T.List[str], shuffle: bool = False):
  nlen = len(seqs[0])
  seqarys = [sequence2array(x) for x in seqs
             if len(x) == nlen]
  if shuffle: random.shuffle(seqarys)
  return seqarys

def encoder(a3m_lines: T.List[str]) -> T.List[np.ndarray]:
  """
  each elements denote all the lines from a a3m file
  0, 2, 4, 6, ... is the sequence names
  1, 3, 5, 7, ... is the sequence content
  """
  # seqs = [line.translate(NOLOWER).rstrip()
  #         for line in a3m_lines
  #         if not (line.startswith(">") or "\x00" in line)]
  seqs = [a3m_lines[i].translate(NOLOWER).rstrip()
          for i in range(1, len(a3m_lines),2)]
  # m = np.array([sequence2array(x) for x in seqs])
  m = np.array(build_array(seqs))
  return MAPPING[m]

class MSAEncoder(nn.Module):
    """
    """
    num_embeddings = 21
    def __init__(self, embedding_dim : int, 
                 encoding_strategy : str = "emb_plus_one_hot"):
        super(MSAEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoding_strategy = encoding_strategy
        self.emb_layer = nn.Embedding(num_embeddings=self.num_embeddings, 
                                  embedding_dim=self.embedding_dim)

        assert encoding_strategy in ["one_hot", "emb", 
                                     "emb_plus_one_hot"],\
            f"the encoding strategy {encoding_strategy} is not implemented"
        
    
    def forward(self, input : th.Tensor) -> th.Tensor:
        x: th.Tensor
        if self.encoding_strategy == "one_hot":
            x = F.one_hot(input, num_classes=self.embedding_dim)
        else:
            x = self.emb_layer(input)

        if self.encoding_strategy == "emb_plus_one_hot":
            return x + F.one_hot(input, num_classes=self.embedding_dim)
        else:
            return x.float()
        
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: T.Optional[int] = 1):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        assert isinstance(self.padding_idx, int)
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

if __name__ == "__main__":
  import time 
  import torch.multiprocessing as mp
  import argparse as P

  mp.set_sharing_strategy("file_system")
  parser = P.ArgumentParser()

  parser.add_argument("dataset")
  parser.add_argument("msa")

  parser.add_argument("-m", "--mode")
  parser.add_argument("-t", "--task")
  parser.add_argument("-c", "--num-classes", type=int)
  parser.add_argument("-s", "--msa-buffer-size", type=int, default=2000)
  parser.add_argument("-l", "--MAXLEN",type=int, default=1000)
  parser.add_argument("-b", "--batch-size", type=int, default=32)
  parser.add_argument("-e", "--epochs", type=int, default=1)
  parser.add_argument("--max-msa-size", type=int, default=-1)
  parser.add_argument("--num-workers", type=int, default=10)
  parser.add_argument("--not-shuffle", action="store_true")

  opt = parser.parse_args()
  print(opt)

  msa_dataset = MSADataset(opt.dataset, opt.msa, 
                           opt.mode,
                           opt.task,
                           opt.num_classes,
                           opt.msa_buffer_size,
                           opt.MAXLEN,
                           msa_max_size=opt.msa_max_size)

  msa_loader = td.DataLoader(msa_dataset, opt.batch_size, num_workers=opt.num_workers, 
                             shuffle=not opt.not_shuffle,
                             pin_memory=True)
  Epochs = opt.epochs
  for epoch in range(Epochs):
    st = time.time()
    for i, (X, y) in enumerate(msa_loader):
      ed = time.time()
      print(f"Epoch [{epoch+1}/{Epochs}]: consumed {ed-st}s for item {i}")
      print(f"X shape {X.shape}, y shape {y.shape}")
      print(np.where(y)[0].shape)
      st = time.time()