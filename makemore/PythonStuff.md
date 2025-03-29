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

### Reading a file 

```python
words = open('names.txt', 'r').read().lower()
words = words.splitlines()
```

### Stoi implementation

```python
chars = sorted(list(set(''.join(words))))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for c, i in stoi.items() }
```

```python
stoi
```

```python
itos
```

### Understanding Multinomial in torch

```python
import torch
```

```python
weights = torch.tensor([0.5, 10, 0.5, 0.5], dtype=torch.float) 
```

```python
torch.multinomial(weights, 2)
```
