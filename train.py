# import basic torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BytePairEncoder,GPT2Config
from tqdm import tqdm
from minbpe.minbpe import BasicTokenizer

batch_size = 64
block_size = 64
max_iters = 3000
eval_interval= 300
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 512
n_heads = 8
n_layer = 3
dropout = 0.2

# torch.manual_seed(42)

with open('shakespeare.txt','r',encoding='utf-8') as f:
    text = f.read()

vocab_size = 500



def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ["train","test"]:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             x = get_batch(split)
#             logits = model(x.to(device))
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


# Transformer Class
class SwiGLU(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        hidden_size = int(n_embed * (4 * (2/3)))  # Corrected calculation
        self.w = nn.Linear(n_embed, hidden_size, bias=False)
        self.v = nn.Linear(n_embed, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, n_embed, bias=False)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.w2(F.silu(self.w(x)) * self.v(x)))
        return out

class Head(nn.Module):
    def __init__(self, head_size: int, n_embed: int, dropout: float):
        super().__init__()
        self.qkv = nn.Linear(n_embed, 3 * head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.qkv.weight)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if training else 0.0, is_causal=True
        )
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embed: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_embed, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, training: bool) -> torch.Tensor:
        out = torch.cat([head(X, training) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, dropout: float):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, n_embed // n_heads, n_embed, dropout)
        self.ffwd = SwiGLU(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.ln1(x)
        x = x + self.sa_heads(x, training)
        x = self.ln2(x)
        return x + self.ffwd(x)

class TransformerDecoder(nn.Module):
    def __init__(self, config :GPT2Config):
        super().__init__()
        self.embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.positon_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config.n_embed, config.n_heads, config.dropout) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        # Weight Tying
        self.embedding_table.weight = self.lm_head.weight

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        B, T = x.shape
        tok_emb = self.embedding_table(x)
        pos_emb = self.positon_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, training)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits



if __name__ == '__main__':
    tokenizer = BasicTokenizer()
    tokenizer.load("tokenizer.model")
    data = torch.tensor(tokenizer.encode(text),dtype=torch.long)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    test_data = data[split:]
    config = GPT2Config(vocab_size=vocab_size,block_size=block_size,n_embed=n_embed,n_heads=n_heads,n_layers=n_layer,dropout=dropout,lr=learning_rate,t_max=max_iters)
    model = TransformerDecoder(config).to(device)
    print("Total Params:",sum([p.numel() for p in model.parameters()]))
    print("Using",device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_iters)

    for iter in tqdm(range(max_iters+1)):
        x,y = get_batch("train")
        logits = model(x)
        logits = logits.view(-1,logits.shape[-1])
        y = y.view(-1)
        loss = F.cross_entropy(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if iter % eval_interval == 0:
            with torch.inference_mode():
                model.eval()
                x,y = get_batch("test")
                logits = model(x)
                logits = logits.view(-1,logits.shape[-1])
                y = y.view(-1)
                loss = F.cross_entropy(logits,y)
                tqdm.write(f"iter {iter} : loss = {loss.item():.4f}")
                model.train()

    torch.save(model.state_dict(),"model.pt")