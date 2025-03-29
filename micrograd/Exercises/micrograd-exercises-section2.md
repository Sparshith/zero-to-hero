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
# Value class starter code, with many functions taken out
from math import exp, log

class Value:
  
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other): # exactly as in the video
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
    
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # Adding __mul__ to power division
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    # Adding __pow__ to power division
    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad +=  ((other) * (self.data ** (other-1)) ) * out.grad

        out._backward = _backward
        return out

    # Adding __radd__ for error: TypeError: unsupported operand type(s) for +: 'int' and 'Value'     
    def __radd__(self, other):
        return self + other

    # Adding for __truediv__ error: TypeError: unsupported operand type(s) for /: 'Value' and 'Value'
    def __truediv__(self, other):
        return (self * (other ** -1))
    def __neg__(self):
        return self * -1

    # Adding exp for error: AttributeError: 'Value' object has no attribute 'exp'
    # exp returns math.exp as the output
    # d(e**x)/dx = e**x, so the backward pass will have the grad as out.data * out.grad
    def exp(self):
        out = Value(exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # Adding log for error: AttributeError: 'Value' object has no attribute 'log'
    def log(self):
        out = Value(log(self.data), (self, ), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out

  
  # ------
  # re-implement all the other functions needed for the exercises below
  # your code here
  # TODO
  # ------

    def backward(self): # exactly as in video  
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()
```

```python
# without referencing our code/video __too__ much, make this cell work
# you'll have to implement (in some cases re-implemented) a number of functions
# of the Value object, similar to what we've seen in the video.
# instead of the squared error loss this implements the negative log likelihood
# loss, which is very often used in classification.

# this is the softmax function
# https://en.wikipedia.org/wiki/Softmax_function
def softmax(logits):
  counts = [logit.exp() for logit in logits]
  denominator = sum(counts)
  out = [c / denominator for c in counts]
  return out

# this is the negative log likelihood loss function, pervasive in classification
logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
probs = softmax(logits)
loss = -probs[3].log() # dim 3 acts as the label for this input example
loss.backward()
print(loss.data)

ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
for dim in range(4):
  ok = 'OK' if abs(logits[dim].grad - ans[dim]) < 1e-5 else 'WRONG!'
  print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad}")

```

```python

```
