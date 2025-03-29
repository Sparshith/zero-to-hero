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
    display_name: Python (micrograd)
    language: python
    name: your_env_name
---

### Call function

Usage: Use when you want to expose the object as a function 

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, number):
        return number * self.factor
        
```

```python
double = Multiplier(2.0)
```

```python
double(4)
```

### Zip function

Usage: Use it when you have two lists, and you want to run some operation on elements on the same index across both lists

```python
x = [1, 2, 4]
y = [10, 20, 30]
zipped = zip(x, y)
```

```python
list(zipped)
```

### Iterating

```python
a = [1, 2, 3, 4]
```

```python
for i in range(0, len(a)):
    print(a[i])
```

```python

```
