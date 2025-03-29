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
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value (data = {self.data})"

    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)
        
```

```python
a = Value(5.0)
```

```python
a
```

```python
b = Value(4.0)
```

```python
a + b
```

```python
a * b 
```

```python

```

```python

```
