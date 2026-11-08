## How Neural Networks Learn

### Forward propagation

Pass the inputs through the model.

```python
prediction = model(inputs)
```

The model produces an output.

### Loss

how wrong the prediction is.

A smaller loss generally means predictions fit the training labels better.

### Backpropagation

how the loss changes with respect to each trainable parameter.

### weights

A small learning rate usually produces smaller parameter updates.

A very large learning rate can make training unstable.