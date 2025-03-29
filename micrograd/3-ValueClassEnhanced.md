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

```python
class Value:
    def __init__(self, data, _children = (), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value (data = {self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '-')
        
```

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
```

```python
d = a * b + c
```

```python
d._prev
```

```python
d._op
```

```python
d
```

```python

```
