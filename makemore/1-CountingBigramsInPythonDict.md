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
words = open('names.txt', 'r').read().splitlines()
```

```python
w = words[0]
```

```python
w[1:]
```

```python
list(w)
```

```python
# Neat trick to generate tuples of cur char, next char
# zipping of w will start from 0, and w[1:] will start from the first index

for c1, c2 in zip(w, w[1:]):
    print(c1, c2)
```

```python
# Need to count number of occurences of each biagram

bigrams = {}
for w in words:
    w = ['<S>'] + list(w) + ['<E>']
    for c1, c2 in zip(w, w[1:]):
        k = c1 + c2
        bigrams[k] = bigrams.get(k, 0) + 1
```

```python
sorted(bigrams.items(), key=lambda kv: -kv[1])
```

```python


```
