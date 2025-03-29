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

### Objective

Create a single number to track the quality of the model. 

This is done via calculating the negative log likelihood of the training data. 

##### Why negative log likelihood?
1. Probability of abc as a string to be generated = product of probability of each bigrams acc to the chain rule, i.e, `P(.abc.) =  P(.a) * P(ab)* P(bc) * P(c.)`
2. Now, each of these probabilities are going to be a number between 0 and 1, so it will likely result in a very low number. To avoid this, we use a monotonic function like log to help scale the value to a meaningful number.
3. Therefore, loss can be `log(P(.abc.)`. Additionally, due to the log product rule, `log(P(.abc.) = log(P(.a)) + log(P(ab)) + log(P(bc) + P(c.))`
4. However, the output of this will be a negative number, so in order to model the loss as a number to reduce to make the model better, we can just take the negative value of this. This is how we end up with the "Negative log likelihood" as the loss fn. Can average the final value by the total number of bigrams to normalise to get a smaller number instead of a large value. 

##### Steps to implement:
1. For each word in the training set, calculate the log probability of each bigram in the loop counting bigrams.
2. Sum up each of them to get the loss function across the entire training set, and avg them by dividing this sum by the total number of bigrams. 

```python
import torch
import math
```

```python
N = torch.zeros((27, 27), dtype=torch.int32)
words = open('names.txt', 'r').read().lower().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {c:i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i:c for c, i in stoi.items()}
```

```python
for w in words:
    w = ['.'] + list(w) + ['.']
    for c1, c2 in zip(w, w[1:]):
        N[stoi[c1]][stoi[c2]] += 1
```

```python
sumOfEachRow = torch.sum(N.float() + 1, dim=1, keepdim=True)
P = N/sumOfEachRow
```

### Calculate the loss

```python
log_likelihood = 0.0
n = 0
for w in words:
    w = ['.'] + list(w) + ['.']
    for c1, c2 in zip(w, w[1:]):
        row = stoi[c1]
        col = stoi[c2]
        log_prob = torch.log(P[row][col]) # log of prob of bigram
        log_likelihood += log_prob
        n += 1
nll = -log_likelihood
```

```python
loss = nll/n
```

```python
loss
```

```python

```
