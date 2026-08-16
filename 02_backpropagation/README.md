# Backpropagation

Understanding how neural networks learn by calculating gradients, optimizing parameters, and transforming network outputs.

## Topics

* Chain rule
* Gradient descent
* Backpropagation
* Gradients
* Parameter optimization
* ReLU activation
* Multiple inputs and outputs
* Logits
* Softmax
* ArgMax
* Softmax Jacobian / derivative
* Cross-entropy loss
* Training over multiple epochs

## Learning Progression

StatQuest → handwritten notes → PyTorch implementation → experiments

## Status

* [x] Chain rule
* [x] Gradient descent
* [x] Backpropagation
* [x] Gradients and parameter optimization
* [x] ReLU
* [x] Multiple inputs and outputs
* [x] Logits
* [x] Softmax and ArgMax
* [x] Softmax derivative / Jacobian
* [ ] Cross-entropy loss
* [ ] Cross-entropy derivatives
* [ ] Complete classification training loop

## Implementations

Current experiments cover:

* Building a neural network with multiple output neurons
* Producing logits for multiple classes
* Converting logits into probabilities using Softmax
* Selecting predicted classes using ArgMax
* Exploring the Softmax Jacobian using `torch.autograd.functional.jacobian`

The goal is to understand what happens inside the classification pipeline rather than treating PyTorch's high-level APIs as black boxes.
