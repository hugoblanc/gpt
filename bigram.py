from typing import List
from torch.nn import functional as F
import torch.nn as nn
import torch


# Hyper params
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter = 5000
eval_iters = 500
n_embd = 384
learning_rate = 3e-4
n_layer = 6
n_head = 6
torch.manual_seed(1337)
batch_size = 32
block_size = 8
dropout = 0.2

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of the text is: ", len(text))
print(text[:1000])


chars = sorted(list(set(text)))
vocab_size = len(chars)
print('total chars:', vocab_size)
print("".join(chars))


string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_String = {i: ch for i, ch in enumerate(chars)}


def encode(string: str): return [string_to_int[char] for char in string]


def decode(integers: List[int]): return ''.join(
    [int_to_String[i] for i in integers])


print(encode('hello'))
print(decode([40, 41, 42, 42, 43]))
print(decode(encode('hello')))


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batchs(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    rand_i = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in rand_i])
    y = torch.stack([data[y+1:y+1+block_size] for y in rand_i])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batchs(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


source, target = get_batchs('train')
print(source.shape)
print(source)

print(target.shape)
print(target)


class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size) -> None:
        super().__init__()
        self.dropout = nn.Dropout()
        self.key: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)

        out = wei @ v
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # (Batch, Time, Channel)   4 , 8, vocab_size (65)
        tok_emb: nn.Embedding = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel().to(device)
out, loss = model(source, target)
print(out.shape)
print(loss)


starting_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(starting_idx, 100)[0].tolist()))


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


iteration = 0
for steps in range(max_iter):
    iteration += 1
    xb, yb = get_batchs('train')

    logits, loss = model(xb, yb)

    if (iteration % eval_iters) is 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(model.generate(starting_idx, 500)[0].tolist()))
