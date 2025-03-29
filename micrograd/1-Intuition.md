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
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

#### Define a function

```python
def f(x):
    return 3*x**2 - 4*x + 5
```

#### Define a set of points x

```python
xs = np.arange(-5, 5, 0.25)
```

```python
ys = f(xs)
```

```python
plt.plot(xs, ys)
```

```python
h = 0.00001
x = 2
```

```python
(f(x + h) - f(x)) / h
```

```python
a = 2.0
b = -3.0
c = 10.0
```

```python
d1 = a * b + c
b = b + h
d2 = a * b + c
```

```python
print(d1)
print(d2)
print(d2-d1)
print((d2-d1)/h)
```

```python

```
