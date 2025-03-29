---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%pip install torch
%pip install numpy
%pip install matplotlib
```

```python
import torch
```

```python
N = torch.zeros((27, 27), dtype=torch.int32)
```

```python
words = open('names.txt', 'r').read().lower().splitlines()
```

```python
chars = sorted(list(set(''.join(words))))
stoi = {c:i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i:c for c, i in stoi.items()}
```

```python
for w in words:
    w = ['.'] + list(w) + ['.']
    for c1, c2 in zip(w, w[1:]):
        N[stoi[c1]][stoi[c2]] += 1
```

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
```

### Sampling for the first char

```python
p = N[0].float()
p = p/p.sum()
p
```

```python

```

```python
itos[idx]
```

### Sample chars in a loop till it ends EOL

```python
# lets generate a few words
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    idx = 0
    name = ''    
    while True:
        p = N[idx].float()
        p = p/p.sum()
        sample = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
        idx = sample.item()
        #print(idx)
        if(idx == 0):
            break
        name += itos[idx]
    print(name)
```

```python
name
```
