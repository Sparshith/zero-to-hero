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

### Section 1 

##### Notes:
1. While differentiating the output wrt to a variable, if the term does not contain that variable, then the term has no "sensitivity" to the final output. Hence, it becomes 0 when differentiated.
2. Chain rule while using d(sin(u))/dx = cos(u) * du/dx
3. Symmetric derivative performs better than the forward derivative because it "looks" both ways - what happens if there's a slight increase AND slight decrease and hence results in a better approximation. 

```python
# here is a mathematical expression that takes 3 inputs and produces one output
from math import sin, cos

def f(a, b, c):
  return -a**3 + sin(3*b) - 1.0/c + b**2.5 - a**0.5

print(f(2, 3, 4))
```

```python
# write the function df that returns the analytical gradient of f
# i.e. use your skills from calculus to take the derivative, then implement the formula
# if you do not calculus then feel free to ask wolframalpha, e.g.:
# https://www.wolframalpha.com/input?i=d%2Fda%28sin%283*a%29%29%29

def gradf(a, b, c):
    df_da = (-3 * (a ** 2)) - (0.5 * (a ** -0.5))
    df_db = (3 * cos(3 * b)) + (2.5 * b ** 1.5)
    df_dc = c ** -2
    
    return [df_da, df_db, df_dc] # todo, return [df/da, df/db, df/dc]

# expected answer is the list of 
ans = [-12.353553390593273, 10.25699027111255, 0.0625]
yours = gradf(2, 3, 4)
for dim in range(3):
  ok = 'OK' if abs(yours[dim] - ans[dim]) < 1e-5 else 'WRONG!'
  print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {yours[dim]}")
```

```python
# now estimate the gradient numerically without any calculus, using
# the approximation we used in the video.
# you should not call the function df from the last cell

def f(a, b, c):
    return -a**3 + sin(3*b) - c**-1 + b**2.5 - a**0.5 
    
a = 2
b = 3
c = 4
h = 0.000001

df_da = (f(a+h, b, c) - f(a, b, c,))/h
df_db = (f(a, b+h, c) - f(a, b, c,))/h
df_dc = (f(a, b, c+h) - f(a, b, c,))/h


# -----------
numerical_grad = [df_da, df_db, df_dc] # TODO
# -----------

for dim in range(3):
  ok = 'OK' if abs(numerical_grad[dim] - ans[dim]) < 1e-5 else 'WRONG!'
  print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {numerical_grad[dim]}")
```

```python
# there is an alternative formula that provides a much better numerical 
# approximation to the derivative of a function.
# learn about it here: https://en.wikipedia.org/wiki/Symmetric_derivative
# implement it. confirm that for the same step size h this version gives a
# better approximation.

def f(a, b, c):
    return -a**3 + sin(3*b) - c**-1 + b**2.5 - a**0.5 

a = 2
b = 3
c = 4
h = 0.000001

df_da = (f(a+h, b, c) - f(a-h, b, c))/(2*h)
df_db = (f(a, b+h, c) - f(a, b-h, c))/(2*h)
df_dc = (f(a, b, c+h) - f(a, b, c-h))/(2*h)


# -----------
numerical_grad2 = [df_da, df_db, df_dc] # TODO
# -----------

for dim in range(3):
  ok = 'OK' if abs(numerical_grad2[dim] - ans[dim]) < 1e-5 else 'WRONG!'
  print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {numerical_grad2[dim]}")
```

```python

```
