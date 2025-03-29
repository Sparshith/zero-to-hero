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

### Operations

```python
a = torch.tensor([1,2,3,4,5], dtype=torch.int)
```

```python
(a/a.sum()).shape
```

```python
a = torch.tensor([[1, 2], [2, 4], [3, 6]], dtype=torch.int)
a1 = torch.sum(a, dim=1, keepdim=True)
a1
a1.shape
```

### Looping

```python
a = torch.tensor([[1, 2], [2, 4], [3, 6]])
for row in a:
    print(row[0].item(), row[1].item())
```

### Data Types

```python
a = torch.tensor([[1, 2], [2, 4], [3, 6]]).float()
for row in a:
    print(row[0].item(), row[1].item())
```

### Generator

```python
g = torch.Generator().manual_seed(1)
torch.rand(3, generator=g)
```

### Multinomials

```python
import torch
```

```python
weights = torch.tensor([0.5, 10, 0.5, 0.5], dtype=torch.float) 
```

```python
torch.multinomial(weights, 2)
```
