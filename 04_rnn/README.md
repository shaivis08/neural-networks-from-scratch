# Recurrent Neural Networks (RNNs)

Understanding how recurrent neural networks process sequential data and maintain information through hidden states.

## Topics

* Sequential data
* Time steps
* Batch size
* Input size
* Hidden state
* Hidden size
* RNN recurrence
* `tanh` activation
* Multiple RNN layers
* `rnn_out`
* `h_n`
* Sequence-to-output architecture

## Implementation

Implemented a basic RNN using PyTorch's `nn.RNN`.

The implementation explores:

* Input tensors with shape `(batch_size, sequence_length, input_size)`
* Hidden-state representations with a configurable `hidden_size`
* Stacked RNN layers using `num_layers`
* Outputs at every time step through `rnn_out`
* Final hidden states through `h_n`
* Selecting the final time-step representation
* Passing the final hidden representation through a linear layer to generate predictions

## Key Concepts

* **Batch size** → number of sequences processed simultaneously.
* **Sequence length** → number of time steps in each sequence.
* **Input size** → number of features at each time step.
* **Hidden size** → number of values in the hidden-state representation.
* **`rnn_out`** → hidden outputs from every time step.
* **`h_n`** → final hidden state for each RNN layer.

For a 2-layer RNN with `batch_size = 4` and `hidden_size = 20`:

`h_n.shape = (2, 4, 20)`

The last layer's final hidden state represents the 20-dimensional hidden representation after processing the complete sequence.

## Learning Progression

StatQuest → conceptual understanding → PyTorch implementation → tensor-shape exploration

