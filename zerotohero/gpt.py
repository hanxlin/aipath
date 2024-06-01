import torch

block_size = 8
batch_size = 32

with open('zerotohero/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    texts = f.read()

chars = sorted(list(set(texts)))
stoi = { ch : i for i, ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[e] for e in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(texts), dtype = torch.long)
n = int(0.9 * len(data))
train = data[:n]
eval = data[n:]

def get_batch(split):
    data = train if split == 'train' else eval
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

bx, by = get_batch('train')
print(bx[0], by[0])

