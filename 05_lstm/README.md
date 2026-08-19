# Long Short-Term Memory (LSTM)

Understanding LSTMs conceptually and implementing them in PyTorch to understand how recurrent networks maintain and update information across sequences.

## Topics

* LSTM architecture
* Sequential processing
* Hidden state
* Cell state
* Forget, input, and output gates
* Long-term vs short-term information
* Hidden size
* Batch size
* Sequence length
* Multiple LSTM layers
* `nn.LSTM`
* `lstm_out`, `h_n`, and `c_n`
* Using the final hidden representation for prediction
* Linear output layer
* MSE loss
* Backpropagation through the LSTM
* SGD optimization
* Training over multiple epochs

## Implementation

The notebook implements an LSTM using PyTorch's `nn.LSTM`.

### Model Structure

```text
Input sequence
      ↓
   LSTM
      ↓
Final hidden representation
      ↓
   Linear
      ↓
  Prediction
```

For the example:

```text
Input shape:       [4, 8, 10]
Hidden size:       20
Number of layers:  2
Output size:       5
```

With `batch_first=True`, the input format is:

```text
[batch_size, sequence_length, input_size]
```

## Understanding LSTM Outputs

`nn.LSTM` returns:

```text
lstm_out, (h_n, c_n)
```

* `lstm_out` — hidden outputs for every timestep from the last LSTM layer.
* `h_n` — final hidden state for every LSTM layer.
* `c_n` — final cell state for every LSTM layer.

For two layers, a batch size of four, and hidden size of 20:

```text
h_n.shape = [2, 4, 20]
c_n.shape = [2, 4, 20]
```

The final timestep can be selected from `lstm_out` and passed through the linear layer to produce the prediction.

## Training

The model is trained using:

```text
Forward pass
      ↓
Calculate MSE loss
      ↓
Zero gradients
      ↓
Backpropagation
      ↓
Optimizer step
```

The forward pass is performed inside the epoch loop so that each epoch uses the updated model parameters and creates a fresh computation graph.

## Key Understanding

The important distinction between the LSTM outputs is:

```text
lstm_out → hidden output at every timestep

h_n → final hidden state of every layer

c_n → final cell state of every layer
```

The internal gates and cell-state updates are handled by PyTorch's `nn.LSTM`. The goal of this implementation is to understand what these outputs represent and how an LSTM connects to a prediction layer and training loop.

## Learning Progress

StatQuest theory → PyTorch/Lightning implementation tutorial → independent PyTorch implementation → shape/output experiments → training loop

## Status

* [x] LSTM architecture
* [x] Hidden state
* [x] Cell state
* [x] LSTM input/output shapes
* [x] Hidden size and batch size
* [x] Multiple LSTM layers
* [x] `lstm_out`
* [x] `h_n`
* [x] `c_n`
* [x] Final timestep → Linear prediction
* [x] MSE loss
* [x] Backpropagation
* [x] SGD optimization
* [x] Training over multiple epochs

## Goal

Build an intuitive understanding of how LSTMs carry information through a sequence while becoming comfortable implementing and training recurrent models in PyTorch.

