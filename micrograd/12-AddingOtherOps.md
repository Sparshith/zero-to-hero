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
    name: micrograd
---

```python
import math
```

```python
class Value:
    def __init__(self, data, _children = (), _op='', label='', grad = 0.0):
        self.data = data
        self._prev = set(_children) # used to store which Values were used to get this data
        self._op = _op # which operation was run on the children Values to get this data
        self._backward = lambda: None
        self.label = label # what should it be known as in the graph vis (just the var name usually)
        self.grad = grad # what is the gradient wrt to the final root node

    def __repr__(self):
        return f"Value (data = {self.data})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
        
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Exponent must be an int or float, got {type(other).__name__}")

        out = Value(self.data ** other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad = out.grad * ( other * self.data ** (other - 1))
            
        out._backward = _backward
        
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            return out.data
        out._backward = _backward
        return out
    
    def tanhOld(self):
        x = self.data
        output = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(output, (self, ), 'tanh')


        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        e = (2 * self).exp()
        return (e - 1)/(e + 1)

    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()
            
        
        
```

```python
a = Value(10.0, 'a')
```

```python
a +1
```

```python
a * 1
```

```python
a - 1
```

```python
a ** 2
```

```python
a / 10
```

```python
a.tanhOld()
```

```python
a.tanh()
```

```python

```

```python

```
