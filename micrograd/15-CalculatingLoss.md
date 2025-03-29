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
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
```

```python
import math
import random
```

```python
class Value:
    def __init__(self, data, label='', _children=(), _op='', grad = 0, _backward = lambda: None):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self.grad = grad
        self._backward = _backward

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, _children = (self, other), _op = '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, _children = (self, other), _op= '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -(other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Exponent must be an int or float, got {type(other).__name__}")

        out = Value(self.data ** other, _children=(self,), label=f"**{other}")

        def _backward():
            self.grad += out.grad * ( self.data ** (other - 1))
            
        out._backward = _backward
        
        return out
            
    
    def tanh(self):
        x = (math.exp(2 * self.data) - 1)/ (math.exp(2 * self.data) + 1)
        out = Value(x, _children = (self, ), _op = 'tanh')

        def _backward():
            self.grad += (1 - (out.data ** 2))
        out._backward = _backward
        return out
        
        

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
class Neuron:
    # nin is the number of inputs to the neuron
    def __init__(self, nin):
        # generate weights & bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = random.uniform(-1, 1)

    def __call__(self, x):
        # x will be a [x1, x2..]
        # need to return activation_func((w1x1 + w2x2) + b)
        dp = sum((xi * wi for xi, wi in zip(x, self.w)))
        return (dp + self.b).tanh()
```

```python
class Layer:
    # nin: number of inputs
    # nout: number of neurons
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0] 
```

```python
class MLP:
    # nin: number of inputs
    # nouts: list of layers where each layer will define how many neurons it has
    def __init__(self, nin, nouts):
        # This is to create a list where each pair of elements will represent inputs <> ouputs
        self.sz = [nin] + nouts

    def __call__(self, x):
        for i in range(0, len(self.sz) - 1):
            x = Layer(self.sz[i], self.sz[i+1])(x)
        return x
```

```python
m = MLP(3, [4, 4, 1])
```

```python
# Earlier, we were just focused on one set of inputs resulting in one output
# Now, we will look at a bunch of inputs resulting in a bunch of outputs, and minimise the error rate by reducing loss
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
```

```python
ypred = [m(x) for x in xs]
```

```python
# We need a method to quanitify the loss - using MSE 
loss = sum((ypredi - ysi)**2 for ypredi, ysi in zip(ypred, ys))
```

```python
loss
```

```python
draw_dot(loss)
```

```python

```
