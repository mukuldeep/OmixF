import torch
import torch.nn as nn
from torch.nn import functional as F

from Omixf1Tokenizer import Omixf1Tokenizer


tokenizer = Omixf1Tokenizer(max_length=7)
vocab_size = tokenizer.vocab_size

#hyper params
batch_size = 32 # no of independent seq to process in parallel
block_size = 16 # def 8,  max context len for prediction
max_iters = 200 #15000
eval_interval = 20 #1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20#200
n_embed = 32

print(device)

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias = False)
    self.query = nn.Linear(n_embed, head_size, bias = False)
    self.value = nn.Linear(n_embed, head_size, bias = False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table  = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    #self.sa_head = Head(n_embed)
    # self.sa_heads = MultiHeadAttention(4,n_embed//4)
    # self.ffwd = FeedForward(n_embed)
    self.blocks = nn.Sequential(
        Block(n_embed, n_head = 4),
        Block(n_embed, n_head = 4),
        Block(n_embed, n_head = 4),
    )
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    B,T = idx.shape
    # logits = self.token_embedding_table(idx) # (B,T,C)
    token_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
    x = token_emb + pos_emb
    #x = self.sa_head(x)
    # x = self.sa_heads(x)
    # x = self.ffwd(x)
    x = self.blocks(x)
    logits = self.lm_head(x) # (B,T,C=vocab size)


    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits,loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = 1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
