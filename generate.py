import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BytePairEncoder
from train import TransformerDecoder

batch_size = 64
block_size = 64
max_iters = 3000
eval_interval= 300
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 512
n_heads = 8
n_layer = 3
dropout = 0.2
vocab_size = 500

torch.manual_seed(42)

tokenizer = BytePairEncoder.load_model(vocab_file="vocab.json",merges_file="merges.txt",vocab_size=vocab_size)

model = TransformerDecoder().to(device)

model.load_state_dict(torch.load("model.pt"))

model.eval()
text = "The "

for i in range(512):
    x = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)[:,-64:]
    logits,_ = model(x)
    next_token = torch.argmax(logits[:,-1,:]).cpu()
    text += tokenizer.decode([next_token.item()])

print(text)